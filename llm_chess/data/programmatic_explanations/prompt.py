import random
from typing import List, Tuple

from phrase_banks import initial_think_phrase



# Main external function we'll use to generate our prompts
def generate_data_sample(fen: str, explanations: List[str], final_statement: str, final_move_uci: str) -> Tuple[str, str, str]:
    """  
    Given a board (FEN notation), explanations, and a final evaluation, create a reasoning trace to train a model on.
    """
    sys_prompt = "chess_task_sysprompt.txt"
    user_prompt = f"""Here is a board in a game you're currently playing. I want you to think through some possible moves you could make and how those moves will likely play out. You may find it helpful to roll-out each line assuming your opponent plays near-optimally. You may also find it helpful to consider the value of the final board state after each roll-out.\n\nAfter you think through your various moves, please end by telling me your chosen move (in UCI notation) within answer tags.\n\n{_convert_fen_to_visual(fen)}"""

    model_response = f"""{random.choice(initial_think_phrase)}
<think> {_format_explanations(explanations, final_statement)} </think>

<answer> {final_move_uci} </answer><|eot_id|>"""

    return sys_prompt, user_prompt, model_response


# --------------------------------------------------
# |               Helper Functions                 |
# --------------------------------------------------
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


def _format_explanations(explanations: List[str], final_statement: str) -> str:
    concat_exp = ""

    for exp in explanations:
        concat_exp += exp + "\n\n"

    return concat_exp + final_statement