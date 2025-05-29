import os
import ast
import numpy as np
import pandas as pd

from llm_chess.data.raw.board import convert_board
from llm_chess.prompts.chat_to_prompt import ChatProcessor


# =================================
# Hyperparams
# =================================
CUR_DIR = "llm_chess/data"
MODEL_VERSION = "llama3"
OUTPUT_FOLDER = f"{CUR_DIR}/cleaned/verl_tasks"
TASKS = [
    {"task": "predictmove", "split": "train", "samples": 4096, "data_source": f'{CUR_DIR}/raw/deepmind_data/train_20k.csv'},
    {"task": "predictmove", "split": "eval", "samples": 128, "data_source": f'{CUR_DIR}/raw/deepmind_data/evals_1k.csv'},
]
GENERATOR_ARGS = {
    "min_possible_moves": 3
}


# =================================
# Various tasks and parent process function
# =================================
def process_tasks(tasks, generator_args, output_folder, model_version):
    chat_processor = ChatProcessor(model_version=model_version)
    for task in tasks:
        # First process dataframe
        df = pd.read_csv(task['data_source'])
        df['Move'] = df['Move'].apply(ast.literal_eval)
        df['Win Probability'] = df['Win Probability'].apply(ast.literal_eval)
        df = df.sample(frac=1).reset_index(drop=True)

        # Call function associated w/ task
        task_function = globals()[f"_{task['task']}"]
        task_function(df, chat_processor, task, generator_args, output_folder)
        

def _predictmove(df, chat_processor, task, generator_args, output_folder, board_notation="visual"):
    """
    Given a board in board_notation format and no legal moves provided,
    generate a move to play.
    """
    outputs = []
    for index, row in df.iterrows():
        if len(outputs) >= task['samples']:
            break

        # Extract key parts of the data
        board = row['FEN']
        moveset = row['Move']
        win_probs = row['Win Probability']

        # Get rid of samples w/ hardly any moves (too easy to predict)
        if len(moveset) < generator_args['min_possible_moves']:
            continue

        # Process various parts of the data
        move_prob_dict = dict(zip(moveset, win_probs))
        move_prob_dict = _process_dict(move_prob_dict, mode="normalize", min_cutoff=0.3)
        user_prompt = f"""Below is a chess board from your current game.

{convert_board(board, board_notation)}

You must select the best move from this position and return it within answer tags. Your answer must be formatted as <answer> my_move </answer>, where my_move is a legal move in UCI notation.

Think step by step if necessary, but do not omit the answer tags or UCI format. Only answers in the correct format will be accepted."""

        data = {
            "data_source": f"chess_{task['task']}",
            "prompt": [
                {
                    "role": "system",
                    "content": chat_processor.get_prompt("chess_task_sysprompt.txt"),
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "ability": "chess",
            "reward_model": {"style": "rule", "ground_truth": str(move_prob_dict)},
            "extra_info": {
                "split": task['split'],
                "data_source": task['data_source']
            },
        }
        outputs.append(data)

    # Export as parquet
    pqt_filename = f"{task['task']}_{len(outputs)}.parquet"
    split_dir = os.path.join(output_folder, task['split'])
    os.makedirs(split_dir, exist_ok=True)
    output_path = os.path.join(split_dir, pqt_filename)
    pd.DataFrame(outputs).to_parquet(output_path)
    print(f"Saved {len(outputs)} samples to {output_path}")


def _process_dict(score_dict, mode="normalize", min_cutoff=0.3):
    """
    Process a {move: score} dict and return a new dict with transformed scores.

    Modes:
        - "normalize": min-max scale to [0, 1]
        - "linear": rank scores, assign linearly spaced values in [0, 1], 1 for best move

    min_cutoff: all values below this threshold (after scaling) are set to 0.
    """
    moves, values = zip(*score_dict.items())
    values = np.asarray(values, dtype=np.float64)

    if mode == "normalize":
        vmin, vmax = values.min(), values.max()
        rng = vmax - vmin
        processed = (values - vmin) / rng if rng else np.ones_like(values)

    elif mode == "linear":
        order = np.argsort(-values)  # descending
        linear_scores = np.linspace(1, 0, num=len(values))
        processed = np.empty_like(values)
        processed[order] = linear_scores

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # Apply min_cutoff: set anything below threshold to 0
    processed = np.where(processed < min_cutoff, 0, processed)

    return dict(zip(moves, processed))


# =================================
# Main loop
# =================================
if __name__ == "__main__":
    process_tasks(
        tasks=TASKS,
        generator_args=GENERATOR_ARGS,
        output_folder=OUTPUT_FOLDER,
        model_version=MODEL_VERSION
    )