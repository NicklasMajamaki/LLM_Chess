import chess
import chess.engine
import random
import numpy as np

def get_n_best_moves_uci_and_evaluation(fen, n, stockfish_path="stockfish", depth=15):
    """
    This function takes a FEN string, a number of plies (n), the path to the Stockfish engine, and a search depth.
    It returns a UCI-style move sequence, the final evaluation of the position,
    and the final FEN string after n plies.
    :param fen: FEN string representing the initial position
    :param n: Number of plies (half-moves) to calculate
    :param stockfish_path: Path to the Stockfish engine executable
    :param depth: Search depth for Stockfish
    :return: A tuple containing the UCI-style move sequence, final evaluation, and final FEN string
    """
    board = chess.Board(fen)
    move_sequence_uci = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:

        # Get the next n plies using Stockfish at a depth of DEPTH
        for i in range(n):
            if board.is_game_over():
                break

            result = engine.analyse(board, chess.engine.Limit(depth=depth))
            best_move = result["pv"][0]
            uci = best_move.uci()
            move_sequence_uci.append(uci)
            board.push(best_move)

        # Evaluate the final position
        evaluation = engine.analyse(board, chess.engine.Limit(depth=depth))["score"]
        if evaluation.is_mate():
            eval_str = f"Mate in {evaluation.mate()}"
        else:
            eval_cp = evaluation.white().score()
            eval_str = f"{eval_cp / 100:.2f} (centipawns)"
        
        final_fen = board.fen()

    # Format the UCI-style move sequence (e.g., "e2e4 e7e5 g1f3 b8c6")
    uci_moves = " ".join(move_sequence_uci)

    return uci_moves, eval_str, final_fen

def get_continuation_from_move(board, move, n_remaining, engine, depth):
    true_turn = 'white' if board.turn == chess.WHITE else 'black'
    move_sequence_uci = [move.uci()]
    board.push(move)

    for _ in range(n_remaining - 1):
        if board.is_game_over():
            break
        result = engine.analyse(board, chess.engine.Limit(depth=depth))
        best_move = result["pv"][0]
        move_sequence_uci.append(best_move.uci())
        board.push(best_move)

    evaluation = engine.analyse(board, chess.engine.Limit(depth=depth))["score"]
    if evaluation.is_mate():
        score_obj = evaluation.white() if true_turn == 'white' else evaluation.black()
        if score_obj.is_mate():
            eval_str = f"Mate in {score_obj.mate()}"
        else:
            eval_str = f"{score_obj.score()} (centipawns)"

    else:
        eval_cp = evaluation.white().score() if true_turn == 'white' else evaluation.black().score()
        eval_str = f"{eval_cp / 100:.2f} (centipawns)"

    final_fen = board.fen()

    uci_moves = " ".join(move_sequence_uci)

    return uci_moves, eval_str, final_fen

def get_best_n_continuation(fen, n, stockfish_path="stockfish", depth=15):
    board = chess.Board(fen)
    true_turn = 'white' if board.turn == chess.WHITE else 'black'
    move_sequence_uci = []

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for i in range(n):
            if board.is_game_over():
                break
            result = engine.analyse(board, chess.engine.Limit(depth=depth))
            best_move = result["pv"][0]
            uci = best_move.uci()
            move_sequence_uci.append(uci)
            board.push(best_move)

        evaluation = engine.analyse(board, chess.engine.Limit(depth=depth))["score"]
        if evaluation.is_mate():
            eval_str = f"Mate in {evaluation.mate()}"
        else:
            eval_cp = evaluation.white().score() if true_turn == 'white' else evaluation.black().score()
            eval_str = f"{eval_cp / 100:.2f} (centipawns)"
        
        final_fen = board.fen()

    uci_moves = " ".join(move_sequence_uci)

    return uci_moves, eval_str, final_fen

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

        if gap < 20:
            num_samples = 4
        elif gap < 50:
            num_samples = 3
        else:
            num_samples = 2

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

def multipv_sampling_from_board(fen, n, max_samples=5, stockfish_path="stockfish", depth=15, temperature=1.0):
    """
    Always includes the best move, then samples the rest according to a softmax distribution over the remaining moves.
    Generates greedy Stockfish continuations for each selected first move.
    """
    all_continuations = []
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        info_list = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=15)
        move_scores = []
        
        for info in info_list:
            move = info["pv"][0]
            score = info["score"]
            
            # Get the score from the perspective of the player to move
            pov_score = score.relative
            
            if pov_score.is_mate():
                eval_score = 100000 if pov_score.mate() > 0 else -100000
            else:
                eval_score = pov_score.score()
            
            move_scores.append((move, eval_score))
        
        # Sort by evaluation to get best move (always highest to lowest since we're using relative scores)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_move = move_scores[0][0]
        
        # Always include best move
        selected_first_moves = [best_move]
        
        # Remove best move before softmax sampling
        remaining_moves_scores = move_scores[1:]
        
        # Calculate the number of samples to take based on how close the top two moves are
        gap = move_scores[0][1] - move_scores[1][1] if len(move_scores) > 1 else 0
        num_samples = max_samples
        
        if gap < 40:
            num_samples = 6
        elif gap < 80:
            num_samples = 5
        elif gap < 120:
            num_samples = 4
        else:
            num_samples = 3
        
        if remaining_moves_scores and num_samples > 1:
            remaining_moves, remaining_scores = zip(*remaining_moves_scores)
            
            # Calculate probabilities with softmax
            probs = softmax(remaining_scores, temperature=temperature)
            
            # Count non-zero probabilities
            non_zero_count = sum(p > 1e-10 for p in probs)
            
            # Adjust num_to_sample if there aren't enough non-zero probabilities
            num_to_sample = min(num_samples - 1, len(remaining_moves), non_zero_count)
            
            if num_to_sample > 0:
                sampled_indices = np.random.choice(
                    len(remaining_moves), 
                    size=num_to_sample, 
                    replace=False, 
                    p=probs
                )
                sampled_moves = [remaining_moves[i] for i in sampled_indices]
                selected_first_moves.extend(sampled_moves)
        
        # For each selected first move, generate greedy Stockfish continuation
        for first_move in selected_first_moves:
            board_copy = chess.Board(fen)
            board_data = get_continuation_from_move(board_copy, first_move, n, engine, depth)
            all_continuations.append(board_data)
            
    return all_continuations

def softmax(scores, temperature=1.0):
    """
    Compute softmax values for each score with temperature scaling.
    Higher temperature makes distribution more uniform, lower makes it more peaked.
    Added guards against numerical overflow.
    """
    import numpy as np
    
    # Apply temperature scaling
    scaled_scores = np.array(scores) / temperature
    
    # Shift scores to prevent overflow
    max_score = np.max(scaled_scores)
    exp_scores = np.exp(scaled_scores - max_score)
    
    # Calculate softmax probabilities
    probs = exp_scores / np.sum(exp_scores)
    
    # Ensure probabilities sum to 1
    probs = probs / np.sum(probs)
    
    return probs

if __name__ == "__main__":
    # Example FEN for testing purposes. In practice we would get these from a dataset.
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"
    n = 5 # Number of plies to calculate
    depth = 19 # Search depth for Stockfish
    max_samples = 5 # Maximum number of samples to take
    stockfish_path = "stockfish"

    print("Original FEN:")
    print(fen)

    # Get best move continuation
    uci_moves, final_eval, final_fen = get_best_n_continuation(fen, n, stockfish_path, depth)
    print("\n▶ Best move continuation:")
    print(uci_moves)
    print(f"Final evaluation after {n} plies: {final_eval}")

    # Get NUM_SAMPLES continuations where the best continuation is always included in the sampled moves, and is always the first element
    print("SAMPLES START:")
    #print(sample_continuations_from_board(fen, n, max_samples=max_samples, stockfish_path=stockfish_path, depth=depth, temperature=1.0))
    print("SAMPLES END")

    print("MULTIPV SAMPLES START:")
    print(multipv_sampling_from_board(fen, n, max_samples=max_samples, stockfish_path=stockfish_path, depth=depth, temperature=1.0))
    print("MULTIPV SAMPLES END")