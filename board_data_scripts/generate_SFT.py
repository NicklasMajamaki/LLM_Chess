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

stockfish_path = "stockfish"
depth = 20
max_samples = 5
temperature = 40.0
continuation_depth = 5

#######################################################
# Assume we have a function to get FEN from a dataset #
#######################################################


def load_prompt_template(template_path):
    """Load the prompt template from a text file"""
    with open(template_path, 'r') as file:
        template = file.read()
    return template

def process_board(board_data, prompt_template):
    """Process a single chess board to get information for SFT"""
    
    board = chess.Board(board_data)
    board_visual = _convert_fen_to_visual(board_data)
    
    # Get trajectories and evaluations using your existing code
    #continuations = sample_continuations_from_board(board_data, continuation_depth, max_samples=max_samples, stockfish_path=stockfish_path, depth=depth, temperature=1.0)
    continuations = multipv_sampling_from_board(board_data, n=5, max_samples=max_samples, stockfish_path=stockfish_path, depth=depth, temperature=40.0)
    
    # Determine best move (assuming our code can identify this)
    best_move = continuations[0][0][:4]
    print(f"Best move: {best_move}")
    
    # Format trajectories for prompt insertion
    trajectories = [t[0] for t in continuations]
    evals = [t[1] for t in continuations]
    formatted_trajectories = "\n".join([f"Line {i+1}: {traj}" for i, traj in enumerate(trajectories)])
    
    # Generate prompt using the template
    # We might want to consider including who's turn it is so the prompt can have something like "figure out the best move for black"
    # or "figure out the best move for white"
    # prompt = prompt_template.format(
    #     board=board_visual,
    #     trajectories=formatted_trajectories
    # )
    
    # Create a complete record
    record = {
        "board": board_visual,
        "trajectories": trajectories,
        "evaluations": evals,
        "best_move": best_move,
        # Add any other metadata we want
    }
    
    return record

def create_parquet_batch(boards_batch, prompt_template, output_file):
    """Process a batch of boards and save as parquet"""
    
    # Process all boards in the batch
    with ProcessPoolExecutor() as executor:
        records = list(executor.map(
            lambda board: process_board(board, prompt_template), 
            boards_batch
        ))
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(records)
    
    # Save as parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
    
    return len(records)

# def main():
#     # Path to prompt template file
#     template_path = "prompt_template.txt"
    
#     # Check if template exists
#     if not os.path.exists(template_path):
#         # Create a default template if it doesn't exist
#         default_template = """Given the chess position:
# {board}

# Consider these possible move trajectories:
# {trajectories}

# What is the best move in this position and why?"""
        
#         with open(template_path, 'w') as file:
#             file.write(default_template)
#         print(f"Created default prompt template at {template_path}")
    
#     # Load the prompt template
#     prompt_template = load_prompt_template(template_path)
    
#     # Load your chess board dataset
#     # (Adjust according to actual data format)
#     #chess_boards = pd.read_parquet("your_chess_boards.parquet")

#     chess_boards = [{'fen': 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4'}]
    
#     # Process in batches to manage memory
#     batch_size = 1
#     total_records = 0
#     print(process_board(chess_boards[0], prompt_template))


    
#     """for i in range(0, len(chess_boards), batch_size):
#         batch = chess_boards[i:i+batch_size].to_dict('records')
#         output_file = f"chess_sft_data_batch_{i//batch_size}.parquet"
        
#         records_processed = create_parquet_batch(batch, prompt_template, output_file)
#         total_records += records_processed
        
#         print(f"Processed batch {i//batch_size}: {records_processed} records")"""
    
#     print(f"Total records processed: {total_records}")

# if __name__ == "__main__":
#     main()
def main():
    data_file_path = os.path.join(os.path.dirname(__file__), '../data/raw/train_20k.csv')
    df = pd.read_csv(data_file_path)
    all_results = []

    k = 0
    
    for i, row in df.head(200).iterrows():
    #for i, row in df.iloc[100:300].iterrows():
        if k % 10 == 0:
            print(f"Processing row {k}")
        k += 1
        prompt_template = load_prompt_template("prompt_template.txt")
        fen = row['FEN']
        record = process_board(fen, prompt_template)
        all_results.append(record)

    with open('more_samples_200.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["board", "trajectories", "evaluations", "best_move"])
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)



if __name__ == "__main__":
    main()
