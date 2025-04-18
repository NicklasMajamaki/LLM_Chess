import chess

def get_piece_name_at_location(fen, location):
    board = chess.Board(fen)
    square = chess.parse_square(location)
    piece = board.piece_at(square)
    
    if piece is None:
        return None

    color = 'white' if piece.color == chess.WHITE else 'black'
    name = piece.piece_type  # this gives an int (1-6)
    name_str = chess.piece_name(name)  # maps int to string name like 'bishop'

    return f"{color} {name_str}"


def convert_board(fen: str, board_representation: str) -> str:
    """
    Given an FEN board state, convert to a specific board representation.

    Various board representations:
    - "fen": Identity -- just returns FEN
    - "spaced_fen": FEN with spaces between pieces and empty squares
    - "visual_simple": Simple visual representation laid out in 2D 
    - "visual": Visual representation of board laid out in 2D with columns and ranks
    """
    if board_representation == "fen":
        preamble = "The following is a board provided in Forsyth-Edwards Notation (FEN) where the rank decreases from 8 to 1 and white pieces are uppercase with black pieces being lowercase.\n"
        return preamble + fen
    elif board_representation == "spaced_fen":
        preamble = "The following is a board provided in Forsyth-Edwards Notation (FEN) where the rank decreases from 8 to 1 and white pieces are uppercase with black pieces being lowercase.\n"
        return preamble + _convert_fen_to_spaced_fen(fen)
    elif board_representation == "visual_simple":
        preamble = "The following is a visual representation of a chess board where the rank decreases from 8 to 1 and white pieces are uppercase with black pieces being lowercase.\n"
        return _convert_fen_to_visual_simple(fen)
    elif board_representation == "visual":
        preamble = "The following is a visual representation of a chess board where white pieces are uppercase with black pieces being lowercase.\n"
        return _convert_fen_to_visual(fen)
    else:
        raise ValueError(f"Unknown board representation: {board_representation}")
    
def _convert_fen_to_spaced_fen(fen: str) -> str:
    initial_fen = fen.split(" ")[0]
    board_details = fen[fen.find(" "):]
    return " ".join(initial_fen) + " " + board_details

def _convert_fen_to_visual_simple(fen: str) -> str:
    placement, active, castling, en_passant, halfmove, fullmove = fen.split()
    lines = []

    # 1) Board
    for rank in placement.split('/'):
        row = []
        for c in rank:
            row.extend(['.'] * int(c)) if c.isdigit() else row.append(c)
        lines.append(' '.join(row))

    lines.append('')  # blank line before details

    # 2) Details
    turn = 'White' if active == 'w' else 'Black'
    lines.append(f"- It is {turn}’s turn to move.")

    rights = []
    if 'K' in castling: rights.append('White can castle kingside')
    if 'Q' in castling: rights.append('White can castle queenside')
    if 'k' in castling: rights.append('Black can castle kingside')
    if 'q' in castling: rights.append('Black can castle queenside')
    if rights:
        lines.append(f"- Castling rights: {', '.join(rights)}.")
    else:
        lines.append(f"- No castling rights available.")

    if en_passant != '-':
        lines.append(f"- En passant target square: {en_passant}.")
    else:
        lines.append(f"- No en passant target square.")

    lines.append(f"- Halfmove clock: {halfmove}")
    lines.append(f"- Fullmove number: {fullmove}")

    return '\n'.join(lines)

def _convert_fen_to_visual(fen: str) -> str:
    placement, active, castling, en_passant, halfmove, fullmove = fen.split()
    lines = []

    # 1) Board with '|' on left
    for i, rank in enumerate(placement.split('/')):
        row = []
        for c in rank:
            if c.isdigit():
                row.extend(['.'] * int(c))
            else:
                row.append(c)
        lines.append(f"{8 - i}| " + ' '.join(row))

    # 2) Bottom border of underscores and file labels
    lines.append("   " + ' '.join(['_' for _ in range(8)]))
    lines.append("   " + ' '.join(list("ABCDEFGH")))
    lines.append("")  # blank line before details

    # 3) Natural‑language details
    turn = 'White' if active == 'w' else 'Black'
    lines.append(f"- It is {turn}’s turn to move.")

    rights = []
    if 'K' in castling: rights.append('White can castle kingside')
    if 'Q' in castling: rights.append('White can castle queenside')
    if 'k' in castling: rights.append('Black can castle kingside')
    if 'q' in castling: rights.append('Black can castle queenside')
    if rights:
        lines.append(f"- Castling rights: {', '.join(rights)}.")
    else:
        lines.append("- No castling rights available.")

    if en_passant != '-':
        lines.append(f"- En passant target square: {en_passant}.")
    else:
        lines.append("- No en passant target square.")

    lines.append(f"- Halfmove clock: {halfmove}")
    lines.append(f"- Fullmove number: {fullmove}")

    return '\n'.join(lines)
