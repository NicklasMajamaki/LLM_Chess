import random
import pathlib
from dataclasses import dataclass, field
from typing import Any, List

import chess
import chess.engine

# ─────────────────────────────────────────────────────────────────────────────
# Variation‑tree node
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VariationNode:
    move: chess.Move
    score: int                      # static eval (cp) from ROOT side’s POV
    minimax: int                    # propagated α‑β value
    is_mate: bool = False
    mate_in: int | None = None
    children: List["VariationNode"] = field(default_factory=list)

    def uci(self) -> str:
        return self.move.uci()


class ChessExplainer:
    """
    Pass in a FEN → get natural‑language explanations for ~2 top moves.
    A shallow, pruned α‑β search builds a sparse tree for each move.
    """

    CLEAR_BEST_THRESH = 75          # cp gap = “forced” reply
    KEEP_WINDOW       = 120         # keep variations within this window
    HEAVY_SWAY        = 300         # ≥300 cp → treat as terminal
    MAX_OPP_BRANCH    = 2
    MAX_US_BRANCH     = 2
    TREE_DEPTH        = 6           # ply: our move … opp move
    _INF              = 10_000_000

    # ──────────────────────────────────────────────────────────────
    # Init / teardown
    # ──────────────────────────────────────────────────────────────
    def __init__(
        self,
        engine_path: str | pathlib.Path,
        depth: int = 15,
        multipv: int = 12,
        think_time: float = 0.4,
    ):
        self.config = {
            "engine_path": engine_path,
            "depth": depth,
            "multipv": multipv,
            "think_time": think_time,
        }
        self._engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))

    def close(self) -> None:
        self._engine.close()

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────
    def analyse_position(self, fen: str) -> List[dict[str, Any]]:
        board       = chess.Board(fen)
        root_color  = board.turn                    # our side
        limit       = self._search_limit()
        raw         = self._engine.analyse(board, limit,
                                           multipv=self.config["multipv"])
        structured  = self._structure_analysis(raw, board, root_color)
        if not structured:
            return [{"uci": "0000", "score": 0,
                     "explanation": "Engine produced no analysis."}]

        best_score = structured[0]["score"]

        # top move + ≤1 alternative inside KEEP_WINDOW
        alt_pool   = [ln for ln in structured[1:]
                      if abs(best_score - ln["score"]) <= self.KEEP_WINDOW]
        to_explain = [structured[0]] + random.sample(alt_pool,
                                                     k=min(1, len(alt_pool)))

        explained: list[dict[str, Any]] = []
        for info in to_explain:
            tree = self._build_tree(board,
                                    info["move"],
                                    ply_left=self.TREE_DEPTH - 1,
                                    is_our_turn=False,          # opp replies next
                                    root_color=root_color,
                                    alpha=-self._INF,
                                    beta=self._INF)
            from move_explanation import MoveExplanation            # break cycle
            explainer = MoveExplanation(board, info, best_score, self, tree)
            explainer.analyse()
            explained.append({
                "uci": info["move"].uci(),
                "score": info["score"],
                "explanation": explainer.explanation,
            })
        return explained

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────
    def _search_limit(self) -> chess.engine.Limit:
        return (chess.engine.Limit(time=self.config["think_time"])
                if self.config["think_time"] > 0
                else chess.engine.Limit(depth=self.config["depth"]))

    def _structure_analysis(
        self,
        raw_list: list[dict],
        board: chess.Board,
        perspective: chess.Color,
    ) -> list[dict]:
        """
        Convert raw engine output so every score is from *perspective* POV.
        """
        out = []
        for entry in sorted(raw_list, key=lambda d: d.get("multipv", 0)):
            score_obj = entry["score"].pov(perspective)
            is_mate, mate_in = score_obj.is_mate(), None
            if is_mate:
                mate_in = score_obj.mate()
                score   = 10_000 if mate_in > 0 else -10_000
            else:
                score = score_obj.score()
            pv = entry.get("pv") or [entry["move"]]
            out.append({
                "move": pv[0],
                "score": score,
                "pv": pv,
                "is_mate": is_mate,
                "mate_in": mate_in,
            })
        # sort so “better for us” comes first
        out.sort(key=lambda d: d["score"], reverse=True)
        return out

    # ──────────────────────────────────────────────────────────────
    # Variation‑tree construction (α‑β with pruning)
    # ──────────────────────────────────────────────────────────────
    def _build_tree(
        self,
        root_board: chess.Board,
        root_move: chess.Move,
        *,
        ply_left: int,
        is_our_turn: bool,
        root_color: chess.Color,
        alpha: int,
        beta: int,
    ) -> VariationNode:
        """
        Sparse α‑β search:
          • Early terminal if |score| ≥ HEAVY_SWAY or depth exhausted
          • ≤ MAX_*_BRANCH replies inside KEEP_WINDOW
          • Forced‑move detection via CLEAR_BEST_THRESH
        """
        # make the move
        board = root_board.copy(stack=False)
        board.push(root_move)

        # evaluate current position
        limit = self._search_limit()
        raw   = self._engine.analyse(board, limit,
                                     multipv=self.config["multipv"])
        lines = self._structure_analysis(raw, board, root_color)
        if not lines:
            static = self._eval_leaf(board, root_color)
            return VariationNode(root_move, static, static)

        static = lines[0]["score"]

        # heavy swing? stop here
        if ply_left == 0 or board.is_game_over() \
           or abs(static) >= self.HEAVY_SWAY:
            return VariationNode(root_move, static, static,
                                 is_mate=lines[0]["is_mate"],
                                 mate_in=lines[0]["mate_in"])

        # candidate replies within KEEP_WINDOW
        cand = [ln for ln in lines
                if abs(ln["score"] - static) <= self.KEEP_WINDOW]

        # branch cap & forced‑move logic
        branch_cap = self.MAX_US_BRANCH if is_our_turn else self.MAX_OPP_BRANCH
        if (not is_our_turn and len(cand) > 1
                and abs(cand[0]["score"] - cand[1]["score"]) > self.CLEAR_BEST_THRESH):
            branch_cap = 1
        cand = cand[:branch_cap]

        # α‑β recursion
        best_val = -self._INF if is_our_turn else self._INF
        node = VariationNode(root_move, static, static,
                             is_mate=lines[0]["is_mate"],
                             mate_in=lines[0]["mate_in"])

        for ln in cand:
            child = self._build_tree(
                board,
                ln["move"],
                ply_left=ply_left - 1,
                is_our_turn=not is_our_turn,
                root_color=root_color,
                alpha=alpha,
                beta=beta,
            )
            node.children.append(child)

            if is_our_turn:
                best_val = max(best_val, child.minimax)
                alpha    = max(alpha, best_val)
            else:
                best_val = min(best_val, child.minimax)
                beta     = min(beta, best_val)

            if beta <= alpha:      # α‑β cut‑off
                break

        node.minimax = best_val
        return node

    # shallow eval used for leaves
    def _eval_leaf(self, board: chess.Board, perspective: chess.Color) -> int:
        score = self._engine.analyse(
            board, chess.engine.Limit(depth=5)
        )["score"].pov(perspective)
        return score.mate() * 10_000 if score.is_mate() else score.score()
