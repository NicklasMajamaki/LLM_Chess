import chess
import chess.engine
import random
import numpy as np

def get_n_best_moves_san_and_evaluation(fen, n, stockfish_path="stockfish", depth=15):
    """
    This function takes a FEN string, a number of plies (n), the path to the Stockfish engine, and a search depth.
    It returns a PGN-style move sequence of the best moves in SAN format, the final evaluation of the position,
    and the final FEN string after n plies.
    :param fen: FEN string representing the initial position
    :param n: Number of plies (half-moves) to calculate
    :param stockfish_path: Path to the Stockfish engine executable
    :param depth: Search depth for Stockfish
    :return: A tuple containing the PGN-style move sequence, final evaluation, and final FEN string
    """
    board = chess.Board(fen)
    move_sequence_san = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:

        # Get the next n plies using Stockfish at a depth of DEPTH
        for i in range(n):
            if board.is_game_over():
                break

            result = engine.analyse(board, chess.engine.Limit(depth=depth))
            best_move = result["pv"][0]
            san = board.san(best_move)
            move_sequence_san.append(san)
            board.push(best_move)

        # Evaluate the final position
        evaluation = engine.analyse(board, chess.engine.Limit(depth=depth))["score"]
        if evaluation.is_mate():
            eval_str = f"Mate in {evaluation.mate()}"
        else:
            eval_cp = evaluation.white().score()
            eval_str = f"{eval_cp / 100:.2f} (centipawns)"
        
        final_fen = board.fen()

    # Format the PGN-style move sequence (e.g., "1. e4 e5 2. Nf3 Nc6")
    pgn_moves = ""
    for i in range(0, len(move_sequence_san), 2):
        move_number = i // 2 + 1
        white_move = move_sequence_san[i]
        black_move = move_sequence_san[i + 1] if i + 1 < len(move_sequence_san) else ""
        pgn_moves += f"{move_number}. {white_move} {black_move} "

    return pgn_moves.strip(), eval_str, final_fen

def get_continuation_from_move(board, move, n_remaining, engine, depth):
    move_sequence_san = [board.san(move)]
    board.push(move)

    for _ in range(n_remaining - 1):
        if board.is_game_over():
            break
        result = engine.analyse(board, chess.engine.Limit(depth=depth))
        best_move = result["pv"][0]
        move_sequence_san.append(board.san(best_move))
        board.push(best_move)

    evaluation = engine.analyse(board, chess.engine.Limit(depth=depth))["score"]
    if evaluation.is_mate():
        eval_str = f"Mate in {evaluation.mate()}"
    else:
        eval_cp = evaluation.white().score()
        eval_str = f"{eval_cp / 100:.2f} (centipawns)"

    final_fen = board.fen()

    pgn_moves = ""
    for i in range(0, len(move_sequence_san), 2):
        move_number = i // 2 + 1
        white_move = move_sequence_san[i]
        black_move = move_sequence_san[i + 1] if i + 1 < len(move_sequence_san) else ""
        pgn_moves += f"{move_number}. {white_move} {black_move} "

    return pgn_moves.strip(), eval_str, final_fen

def get_best_n_continuation(fen, n, stockfish_path="stockfish", depth=15):
    board = chess.Board(fen)
    move_sequence_san = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for i in range(n):
            if board.is_game_over():
                break
            result = engine.analyse(board, chess.engine.Limit(depth=depth))
            best_move = result["pv"][0]
            san = board.san(best_move)
            move_sequence_san.append(san)
            board.push(best_move)

        evaluation = engine.analyse(board, chess.engine.Limit(depth=depth))["score"]
        if evaluation.is_mate():
            eval_str = f"Mate in {evaluation.mate()}"
        else:
            eval_cp = evaluation.white().score()
            eval_str = f"{eval_cp / 100:.2f} (centipawns)"
        
        final_fen = board.fen()

    pgn_moves = ""
    for i in range(0, len(move_sequence_san), 2):
        move_number = i // 2 + 1
        white_move = move_sequence_san[i]
        black_move = move_sequence_san[i + 1] if i + 1 < len(move_sequence_san) else ""
        pgn_moves += f"{move_number}. {white_move} {black_move} "

    return pgn_moves.strip(), eval_str, final_fen


def softmax(x, temperature=1.0):
    x = np.array(x)
    x = x - np.max(x)
    exp_x = np.exp(x / temperature)
    return exp_x / np.sum(exp_x)

def sample_continuations_from_board(fen, n, max_samples=5, stockfish_path="stockfish", depth=15, temperature=1.0):
    """
    Always includes the best move, then samples the rest according to a softmax distribution over the remaining moves.
    Generates greedy Stockfish continuations for each selected first move.
    """
    all_continuations = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        move_scores = []

        # Evaluate all legal first moves
        for move in legal_moves:
            board.push(move)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            board.pop()

            score = info["score"]
            if score.is_mate():
                eval_score = 100000 if score.mate() > 0 else -100000
            else:
                eval_score = score.white().score()
            move_scores.append((move, eval_score))

        # Sort by evaluation to get best move
        maximizing = board.turn == chess.WHITE
        move_scores.sort(key=lambda x: x[1], reverse=maximizing)
        best_move, _ = move_scores[0]

        # Always include best move
        selected_first_moves = [best_move]

        # Remove best move before softmax sampling
        remaining_moves_scores = move_scores[1:]

        # Calculate the number of samples to take based on how close the top two moves are
        gap = move_scores[0][1] - move_scores[1][1] if len(move_scores) > 1 else 0
        num_samples = max_samples

        if gap < 50:
            num_samples = 4
        elif gap < 70:
            num_samples = 3
        elif gap < 100:
            num_samples = 2
        else:
            num_samples = 1

        if remaining_moves_scores and num_samples > 1:
            remaining_moves, remaining_scores = zip(*remaining_moves_scores)

            probs = softmax(remaining_scores, temperature=temperature)
            num_to_sample = min(num_samples - 1, len(remaining_moves))

            sampled_indices = np.random.choice(len(remaining_moves), size=num_to_sample, replace=False, p=probs)
            sampled_moves = [remaining_moves[i] for i in sampled_indices]

            selected_first_moves.extend(sampled_moves)

        # For each selected first move, generate greedy Stockfish continuation
        for first_move in selected_first_moves:
            board_copy = chess.Board(fen)
            board_data = get_continuation_from_move(board_copy, first_move, n, engine, depth)

            all_continuations.append(board_data)

    return all_continuations

if __name__ == "__main__":
    # Example FEN for testing purposes. In practice we would get these from a dataset.
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"
    n = 5 # Number of plies to calculate
    depth = 15 # Search depth for Stockfish
    max_samples = 5 # Maximum number of samples to take
    stockfish_path = "stockfish"

    print("Original FEN:")
    print(fen)

    # Get best move continuation
    pgn_moves, final_eval, final_fen = get_best_n_continuation(fen, n, stockfish_path, depth)
    print("\n▶ Best move continuation:")
    print(pgn_moves)
    print(f"Final evaluation after {n} plies: {final_eval}")

    # Get NUM_SAMPLES continuations where the best first move is always included in the sampled moves
    print("SAMPLES START:")
    print(sample_continuations_from_board(fen, n, max_samples=max_samples, stockfish_path=stockfish_path, depth=depth, temperature=1.0))
    print("SAMPLES END")

