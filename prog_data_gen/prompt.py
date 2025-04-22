def convert_fen_to_visual(fen: str) -> str:
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