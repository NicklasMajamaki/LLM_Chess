import chess
import chess.engine
import random
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from board_formatting import _convert_fen_to_visual
from board_data import sample_continuations_from_board
from board_data import get_best_n_continuation
from board_data import multipv_sampling_from_board
import os
import csv
import ast
import re

def parse_eval_string(eval_str):
    eval_str = eval_str.strip()
    mate_match = re.match(r"Mate in (-?\d+)", eval_str)
    if mate_match:
        mate_value = int(mate_match.group(1))
        # High centipawn equivalent for mate: ±10000 minus number of moves
        if mate_value > 0:
            return 10000 - mate_value
        else:
            return -10000 + abs(mate_value)
    else:
        return float(eval_str.split()[0])

def get_first_moves(trajectory_list):
    return [traj.split()[0] for traj in trajectory_list]

def get_acceptable_answers(row):
    try:
        evals = [parse_eval_string(e) for e in ast.literal_eval(row['evaluations'])]
        trajectories = ast.literal_eval(row['trajectories'])
        moves = get_first_moves(trajectories)
        best_move = row['best_move']

        if best_move in moves:
            best_idx = moves.index(best_move)
            best_eval = evals[best_idx]
        else:
            best_eval = max(evals)

        # Accept moves within 0.3 centipawns of best (or same mate)
        acceptable_moves = [move for move, eval_ in zip(moves, evals)
                            if eval_ >= best_eval - 0.3]

        return acceptable_moves
    except Exception as e:
        print(f"Error in row: {row}\nException: {e}")
        return []

def load_prompt_template(template_path):
    """Load the prompt template from a text file"""
    with open(template_path, 'r') as file:
        template = file.read()
    return template


def main():
    # Load the prompt template
    template_path = "prompt_template.txt"
    prompt_template = load_prompt_template(template_path)

    # Load the dataset
    dataset_path = "high_temp_100.csv"
    df = pd.read_csv(dataset_path)

    df['answer'] = df.apply(get_acceptable_answers, axis=1)

    print("Average answer length:", df['answer'].apply(len).mean())

    df['prompt'] = df.apply(
        lambda row: (
            lambda shuffled_trajs: prompt_template.format(
                board=row['board'],
                trajectories="\n".join(
                    f"{i+1}. {traj}" for i, traj in enumerate(shuffled_trajs)
                )
            )
        )(random.sample(ast.literal_eval(row['trajectories']), k=len(ast.literal_eval(row['trajectories'])))),
        axis=1
    )

    print("Sample prompt:", df['prompt'].iloc[0])
    print("Sample answer:", df['answer'].iloc[0])

    df_out = df[['board', 'prompt', 'answer']]
    df_out.to_parquet('output.parquet', index=False)

    


if __name__ == "__main__":
    main()


