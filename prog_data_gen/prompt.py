import random

from phrase_banks import initial_think_phrase
from typing import List, Tuple



# Main external function we'll use to generate our prompts
def generate_data_sample(fen: str, explanations: List[str], final_statement: str, final_move_uci: str) -> Tuple[str, str, str]:
    """  
    Given a board (FEN notation), explanations, and a final evaluation, create a reasoning trace to train a model on.
    """

    sys_prompt = f"""<|begin_of_text|><|header_start|>system<|header_end|>
You are a chess grandmaster currently playing a very strong opponent. Assume they will be playing optimally. 

Shortly you'll be provided with a board state by the user -- please analyze it and think through your possible moves.

You should refer to moves in UCI notation (e.g., d7d5) and should include your thinking in think tags (e.g., <think> your_thinking </think>) and your answer in answer tags (e.g., <answer> UCI_move </answer>). 

As a technique you may want to consider enumerating possible moves and simulating the likely trajectory that would ensue.<|eot|>"""

    user_prompt_plus_format = f"""<|header_start|>user<|header_end|>
{_convert_fen_to_visual(fen)} <|eot|>
<|header_start|>assistant<|header_end|>"""

    model_response = f"""{random.choice(initial_think_phrase)}
<think>
{_format_explanations(explanations, final_statement)}
</think>

<answer> {final_move_uci} </answer><|eot|>"""

    return sys_prompt, user_prompt_plus_format, model_response


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