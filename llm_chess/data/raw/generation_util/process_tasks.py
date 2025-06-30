import os
import ast
import json
import random
import numpy as np
import pandas as pd

from llm_chess.data.raw.board import convert_board, get_piece_name_at_location
from llm_chess.prompts.chat_to_prompt import ChatProcessor

from .exceptions import DiscardedSample
from .verl_prompts import user_prompt_bank



# ==========================================
# Main functions handling dataset creation
# ==========================================
def process_tasks(tasks, generator_args, output_folder, model_version, output_type="parquet"):
    chat_processor = ChatProcessor(model_version=model_version)
    
    for task in tasks:
        # First process dataframe
        df = pd.read_csv(task['data_source'])
        df['Move'] = df['Move'].apply(ast.literal_eval)
        df['Win Probability'] = df['Win Probability'].apply(ast.literal_eval)
        df = df.sample(frac=1).reset_index(drop=True)   # Shuffle df

        # Create our dataset
        rl_dataset, generation_data = create_rl_dataset(
            df = df,
            chat_processor = chat_processor,
            task = task,
            generator_args = generator_args
        )

        split_dir = os.path.join(output_folder, task['split'])
        os.makedirs(split_dir, exist_ok=True)

        # Export
        if output_type == 'parquet':
            filename = f"{task['type']}.parquet"
            output_path = os.path.join(split_dir, filename)
            pd.DataFrame(rl_dataset).to_parquet(output_path)
        elif output_type == 'jsonl':
            filename = f"{task['type']}.jsonl"
            output_path = os.path.join(split_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in rl_dataset:
                    f.write(json.dumps(item) + '\n')
        elif output_type == 'json':
            filename = f"{task['type']}.json"
            output_path = os.path.join(split_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(rl_dataset, f, indent=2)
        else:
            ValueError(f"Output type undefined for: {output_type}.")

        print(f"Saved {len(rl_dataset)} samples to {output_path}. Gen data: {generation_data}.")


def create_rl_dataset(df, chat_processor, task, generator_args, board_notation="visual"):
    """
    Given a dataframe and desired task, create a parquet dataset in a format required for verl.
    """
    outputs = []
    generation_data = {
        "discarded_samples": 0
    }
    for index, row in df.iterrows():
        if len(outputs) >= task['samples']:
            break
        try:
            sys_prompt, user_prompt, ground_truth = _generate_sample(
                df_row = row, 
                task_type = task['type'], 
                generator_args = generator_args,
                chat_processor = chat_processor,
                board_notation = board_notation
            )
        except DiscardedSample as e:
            generation_data['discarded_samples'] += 1
            continue
        except Exception as e:
            print(f"Unknown error encountered: {e}")
            raise e
            
        data = {
            "data_source": f"chess_{task['type']}",
            "prompt": [
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "ability": "chess",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "board": row["FEN"],
                "split": task['split'],
                "data_source": task['data_source']
            },
        }
        outputs.append(data)

    return outputs, generation_data


# ==========================================
# Various helpers
# ==========================================
def _generate_sample(df_row, task_type, generator_args, chat_processor, board_notation):
    """
    Multi-case function to generate an RL data sample based on a desired task_type and sample from our train / eval dataset.
    """
    board = df_row['FEN']
    moveset = df_row['Move']
    win_probs = df_row['Win Probability']
    move_prob_dict = dict(zip(moveset[:], win_probs[:]))
    move_prob_list = list(zip(moveset[:], win_probs[:]))

    # Generate our datapoints based on the task at hand
    if task_type == "predictmove":
        if len(moveset) < generator_args['predictmove_min_possible_moves']:
            raise DiscardedSample()
        
        sys_prompt = chat_processor.get_prompt("chess_task_sysprompt.txt")
        user_prompt = user_prompt_bank[task_type].format(
            formatted_board = convert_board(board, board_notation)
        )

        ground_truth = str(_score_scaling(
            move_prob_dict, 
            score_scaling = generator_args['predictmove_score_scaling'],
            score_cut = generator_args['predictmove_score_cut']
        ))

    elif task_type == "bestmove":
        # First, generate our ground truth
        best_move, best_move_win_prob = max(move_prob_list, key=lambda x: x[1])
        filtered_moves = [
            move for move, prob in move_prob_list 
            if prob < best_move_win_prob - generator_args['bestmove_move_threshold']
        ]
        if len(filtered_moves) < generator_args['bestmove_provided_moves'] - 1:
            raise DiscardedSample()
        move_candidates = random.sample(filtered_moves, generator_args['bestmove_provided_moves'] - 1)
        move_candidates.append(best_move)
        random.shuffle(move_candidates)
        ground_truth = str({
            "answer": best_move,
            "candidates": move_candidates
        })

        # Then generate our sys and user prompts
        sys_prompt = chat_processor.get_prompt("chess_task_sysprompt.txt")
        user_prompt = user_prompt_bank[task_type].format(
            formatted_board = convert_board(board, board_notation),
            move_candidates = move_candidates
        )

    elif task_type == "worstmove":
        # First, generate our ground truth
        worst_move, worst_move_win_prob = min(move_prob_list, key=lambda x: x[1])
        filtered_moves = [
            move for move, prob in move_prob_list 
            if prob > worst_move_win_prob + generator_args['worstmove_move_threshold']
        ]
        if len(filtered_moves) < generator_args['worstmove_provided_moves'] - 1:
            raise DiscardedSample()
        move_candidates = random.sample(filtered_moves, generator_args['worstmove_provided_moves'] - 1)
        move_candidates.append(worst_move)
        random.shuffle(move_candidates)
        ground_truth = str({
            "answer": worst_move,
            "candidates": move_candidates
        })

        # Then generate our sys and user prompts
        sys_prompt = chat_processor.get_prompt("chess_task_sysprompt.txt")
        user_prompt = user_prompt_bank[task_type].format(
            formatted_board = convert_board(board, board_notation),
            move_candidates = move_candidates
        )

    elif task_type == "legalmoves":
        piece_counts = {}
        for move in moveset:
            piece_pos = move[:2]
            piece_counts[piece_pos] = piece_counts.get(piece_pos, 0) + 1
        valid_pieces = [k for k, v in piece_counts.items() if v >= generator_args['legalmoves_min_moves']]
        if not valid_pieces:
            raise DiscardedSample()
        
        piece_pos = random.choice(valid_pieces)
        piece_name = get_piece_name_at_location(board, piece_pos)
        if piece_name is None:
            print(f"Piece not found at {piece_pos} in FEN: {board}")
            DiscardedSample()

        ground_truth = str([move for move in moveset if move.startswith(piece_pos)])

        sys_prompt = chat_processor.get_prompt("chess_task_sysprompt.txt")
        user_prompt = user_prompt_bank[task_type].format(
            formatted_board = convert_board(board, board_notation),
            piece_name = piece_name,
            piece_pos = piece_pos
        )

    else:
        raise ValueError(f"Invalid task: {task_type}.")

    return sys_prompt, user_prompt, ground_truth



def _score_scaling(score_dict, score_scaling="normalize", score_cut=0.3):
    """
    Process a {move: score} dict and return a new dict with transformed scores.

    Modes:
        - "normalize": min-max scale to [0, 1]
        - "linear": rank scores, assign linearly spaced values in [0, 1], 1 for best move

    min_cutoff: all values below this threshold (after scaling) are set to 0.
    """
    moves, values = zip(*score_dict.items())
    values = np.asarray(values, dtype=np.float64)

    if score_scaling == "normalize":
        vmin, vmax = values.min(), values.max()
        rng = vmax - vmin
        processed = (values - vmin) / rng if rng else np.ones_like(values)
    elif score_scaling == "linear":
        order = np.argsort(-values)  # descending
        linear_scores = np.linspace(1, 0, num=len(values))
        processed = np.empty_like(values)
        processed[order] = linear_scores
    else:
        raise ValueError(f"Unknown mode '{score_scaling}'")    
    
    # Apply min_cutoff: set anything below threshold to 0
    processed = np.where(processed < score_cut, 0, processed)

    return dict(zip(moves, processed.astype(float).tolist()))