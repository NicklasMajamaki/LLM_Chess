import chess
import chess.engine

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

if __name__ == "__main__":
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"
    n = 5  # number of plies (half-moves)
    depth = 15 # Stockfish search depth. It is currently set to 15 for speed but can be increased up to ~23 for better accuracy.
    stockfish_path = "stockfish"

    pgn_moves, final_eval, final_fen = get_n_best_moves_san_and_evaluation(fen, n, stockfish_path, depth)

    print("Original FEN:")
    print(fen)
    print("\nFinal FEN:")
    print(final_fen)

    print("PGN move sequence:")
    print(pgn_moves)
    print(f"\nFinal evaluation after {n} plies: {final_eval}")
