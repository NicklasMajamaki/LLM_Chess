import chess
import chess.engine
import random

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

if __name__ == "__main__":
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"
    n = 5
    depth = 15
    stockfish_path = "stockfish"

    print("Original FEN:")
    print(fen)

    # Get best move continuation
    pgn_moves, final_eval, final_fen = get_best_n_continuation(fen, n, stockfish_path, depth)
    print("\n▶ Best move continuation:")
    print(pgn_moves)
    print(f"Final evaluation after {n} plies: {final_eval}")

    # Evaluate all legal moves for first move quality
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        move_scores = []
        for move in legal_moves:
            board.push(move)
            score = engine.analyse(board, chess.engine.Limit(depth=depth))["score"]
            board.pop()

            if score.is_mate():
                eval_score = 100000 if score.mate() > 0 else -100000
            else:
                eval_score = score.white().score()
            move_scores.append((move, eval_score))

        move_scores.sort(key=lambda x: x[1], reverse=(board.turn == chess.WHITE))

        best_move = move_scores[0][0]
        worst_move = move_scores[-1][0]

        # Select a random move not equal to best or worst
        other_moves = [m for m, _ in move_scores if m != best_move and m != worst_move]
        random_move = random.choice(other_moves) if other_moves else best_move

        # Get continuations
        board_copy = chess.Board(fen)
        worst_continuation, worst_eval, worst_board = get_continuation_from_move(board_copy, worst_move, n, engine, depth)

        board_copy = chess.Board(fen)
        random_continuation, random_eval, random_board = get_continuation_from_move(board_copy, random_move, n, engine, depth)

    print("\n▶ Worst move continuation:")
    print(worst_continuation)
    print(f"Final evaluation after {n} plies: {worst_eval}")

    print("\n▶ Random move continuation (≠ best, ≠ worst):")
    print(random_continuation)
    print(f"Final evaluation after {n} plies: {random_eval}")
