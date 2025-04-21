from __future__ import annotations
import random
import chess
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from chess_explainer import ChessExplainer, VariationNode


class MoveExplanation:
    """Narrates a VariationNode tree like a human would tell a friend."""
    BLUNDER_THRESHOLD = 200                     # cp gap vs engine best

    PIECE_NAMES = {
        chess.PAWN:   "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK:   "rook",
        chess.QUEEN:  "queen",
        chess.KING:   "king",
    }
    # qualitative buckets for evaluation swings
    SWING_TEXT = [
        (300, "looks completely winning for us now"),
        (150, "gives us a very pleasant position"),
        (50,  "tilts things slightly in our favour"),
        (-50, "keeps the balance"),
        (-150, "hands the initiative to the opponent"),
        (-300, "leaves us fighting for survival"),
    ]

    # varied phrasing pools
    OPP_OPEN   = ["Opponent might answer with", "They could reply", "A typical response is",
                  "Black can try", "White may choose"]
    OUR_OPEN   = ["We would continue with", "Our follow‑up is", "Next we play", "Then we go for"]
    VERDICT_PHRASE = ["All in all,", "Overall,", "Big picture,", "Putting it together,"]

    # ──────────────────────────────────────────────────────────────
    def __init__(
        self,
        board_before: chess.Board,
        move_info: Dict[str, Any],
        best_score: int,
        explainer: ChessExplainer,
        variation_root: "VariationNode",
    ):
        self.board_before = board_before
        self.move = move_info["move"]
        self.score = move_info["score"]
        self.is_mate = move_info.get("is_mate", False)
        self.mate_in = move_info.get("mate_in")
        self.best_score = best_score
        self.explainer = explainer
        self.root = variation_root

        self.board_after = board_before.copy(stack=False)
        self.board_after.push(self.move)

        self.explanation: str | None = None

    # ──────────────────────────────────────────────────────────────
    def analyse(self) -> None:
        self.explanation = self._compose()

    # ──────────────────────────────────────────────────────────────
    # Composer
    # ──────────────────────────────────────────────────────────────
    def _compose(self) -> str:
        parts: List[str] = []

        # opening line – our candidate move
        parts.append(f"We start with {self._describe(self.board_before, self.move)}.")

        # narrate each opponent branch (shuffle after best)
        if self.root.children:
            branches = sorted(self.root.children, key=lambda n: n.score)
            if len(branches) > 1:
                best, rest = branches[0], branches[1:]
                random.shuffle(rest)
                branches = [best] + rest
            for idx, child in enumerate(branches):
                parts.extend(self._branch_story(idx, child))

        parts.append(self._verdict())
        return " ".join(parts)

    # ------------------------------------------------------------------
    def _branch_story(self, idx: int, opp: "VariationNode") -> List[str]:
        seq: List[str] = []
        opener = random.choice(self.OPP_OPEN) if idx == 0 \
                 else random.choice(["Alternatively,", "Another possibility is"])

        opp_board = self.board_after.copy(stack=False)
        opp_line = f"{opener} {self._describe(opp_board, opp.move, False)}"
        eff = self._effects(opp_board, opp.move)
        if eff:
            opp_line += f", {eff}"
        opp_line += "."
        seq.append(opp_line)
        opp_board.push(opp.move)

        # our reply, if any
        if opp.children:
            reply = max(opp.children, key=lambda n: n.score)
            our_line = f"{random.choice(self.OUR_OPEN)} {self._describe(opp_board, reply.move, False)}"
            ceff = self._effects(opp_board, reply.move)
            if ceff:
                our_line += f", {ceff}"
            our_line += "."
            seq.append(our_line)
            opp_board.push(reply.move)

            # one more enemy move if present
            if reply.children:
                opp2 = min(reply.children, key=lambda n: n.score)
                seq.append(f"{random.choice(self.OPP_OPEN)} {self._describe(opp_board, opp2.move, False)}.")
                opp_board.push(opp2.move)

        # material & swing summary
        mat = self._material_text(self.board_before, opp_board)
        swing = self._swing_text(opp.minimax - self.score)
        summary_bits = [b for b in [mat, swing] if b]
        if summary_bits:
            seq.append(" ".join(summary_bits) + ".")

        return seq

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    def _verdict(self) -> str:
        prefix = random.choice(self.VERDICT_PHRASE)
        if self.is_mate:
            return f"{prefix} this forces mate in {self.mate_in}."

        gap = self.best_score - self.score
        if gap >= self.BLUNDER_THRESHOLD:
            return f"{prefix} that move is probably a blunder – stronger options exist."
        if gap >= self.BLUNDER_THRESHOLD // 2:
            return f"{prefix} it feels risky and there may be safer choices."
        if abs(gap) <= 20:
            return f"{prefix} perfectly playable – nothing wrong with it."
        return f"{prefix} it should be fine and keeps the game going."

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    def _effects(self, board: chess.Board, move: chess.Move) -> str:
        out = []
        tgt = board.piece_at(move.to_square)
        if tgt:
            out.append(f"picking up a {self.PIECE_NAMES[tgt.piece_type]}")
        tmp = board.copy(stack=False); tmp.push(move)
        if tmp.is_check():
            out.append("giving check")
        if move.promotion:
            out.append(f"promoting to a {self.PIECE_NAMES[move.promotion]}")
        return " and ".join(out)

    def _material_text(self, before: chess.Board, after: chess.Board) -> str | None:
        diff = {pt: len(after.pieces(pt, before.turn)) - len(before.pieces(pt, before.turn))
                for pt in self.PIECE_NAMES}
        plus  = [self.PIECE_NAMES[pt] for pt, d in diff.items() if d > 0]
        minus = [self.PIECE_NAMES[pt] for pt, d in diff.items() if d < 0]
        if plus and not minus:
            return f"we come out a {', '.join(plus)} up"
        if minus and not plus:
            return f"we drop a {', '.join(minus)}"
        if plus and minus:
            return ("material is mixed – we gain "
                    + ", ".join(plus) + " but lose "
                    + ", ".join(minus))
        return None

    def _swing_text(self, cp: int) -> str | None:
        for thresh, phrase in self.SWING_TEXT:
            if cp >= thresh:
                return phrase
            if cp <= -thresh:
                return phrase.replace("us", "them")
        return None

    def _describe(self, board: chess.Board, move: chess.Move, cap: bool = True) -> str:
        piece = board.piece_at(move.from_square)
        if not piece:
            return move.uci()
        name = self.PIECE_NAMES[piece.piece_type]
        if cap:
            name = name.capitalize()
        tgt = board.piece_at(move.to_square)
        sq  = chess.square_name(move.to_square)
        if tgt:
            return f"{name} takes {self.PIECE_NAMES[tgt.piece_type]} on {sq} ({move.uci()})"
        return f"{name} to {sq} ({move.uci()})"
