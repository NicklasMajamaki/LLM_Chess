from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

import chess
import chess.engine

###############################################################################
# Data structures
###############################################################################

@dataclass(slots=True)
class VariationNode:
    """One ply in the variation tree."""

    move: chess.Move
    score: int                     # Static evaluation **after** this move
    minimax: int                   # Result of minimax (+αβ) below this node
    is_mate: bool = False          # Engine says position is forced mate
    mate_in: Optional[int] = None  # Moves until mate (sign ‑ for us to move)
    children: List["VariationNode"] = field(default_factory=list)

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------

    def uci(self) -> str:
        """Return move in UCI notation."""
        return self.move.uci()

    def visualize(self, depth: int = 0) -> str:
        """Pretty‑print subtree for debugging."""
        indent = "  " * depth
        line = f"{indent}{self.uci()} (score={self.score}, minimax={self.minimax})"
        if self.is_mate:
            line += f" (mate in {self.mate_in})"
        for child in self.children:
            line += "\n" + child.visualize(depth + 1)
        return line

###############################################################################
# Main driver
###############################################################################

class ChessExplainer:
    """Light‑weight wrapper around Stockfish to build explanation trees."""

    # Tunables – tweak for speed/quality trade‑off --------------------------
    INITIAL_MOVES_SAMPLED = 3        # How many root moves to explain
    KEEP_WINDOW_CENTI = 200          # Alt moves within this window are kept

    CLEAR_BEST_THRESH = 60           # Δcp to bother exploring 2nd/3rd reply
    MAX_TREE_NODES = 12              # Hard cap to keep engine calls bounded
    MAX_TREE_DEPTH = 3               # Plies *after* the root move
    MAX_BRANCHING = 3                # Reply lines per node (PV + alternates)

    INF = 10_000_000                 # Sentinel for minimax initialisation
    MATE_SCORE = 10_000              # Normalised mate value (≫ any cp score)

    # ------------------------------------------------------------------
    # Construction / teardown
    # ------------------------------------------------------------------

    def __init__(
        self,
        engine_path: str | Path,
        depth: int = 15,
        multipv: int = 8,
        think_time: float = 0.4,
    ) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        self._root_cfg: Dict[str, Any] = {
            "depth": depth,
            "multipv": multipv,
            "think_time": think_time,
        }
        self._nodes_created = 0
        self._root_color: Optional[chess.Color] = None

    # Public helpers --------------------------------------------------------

    def close(self) -> None:
        """Shut down Stockfish subprocess."""
        self._engine.close()

    # ------------------------------------------------------------------
    # High level API
    # ------------------------------------------------------------------

    def analyze_position(self, fen: str) -> List[Dict[str, Any]]:
        """Return a list of explanation trees for *fen* root position."""
        board = chess.Board(fen)
        self._root_color = board.turn

        root_lines = self._analyze(board, **self._root_cfg)
        if not root_lines:
            return []  # No legal moves – nothing to explain.

        # Choose which candidate moves to expand.
        best = root_lines[0]
        alts = [l for l in root_lines[1:] if abs(best["score"] - l["score"]) <= self.KEEP_WINDOW_CENTI]
        chosen = [best] + random.sample(alts, min(len(alts), self.INITIAL_MOVES_SAMPLED - 1))

        trajectories: List[Dict[str, Any]] = []
        for info in chosen:
            self._nodes_created = 0
            tree = self._build_tree(
                board.copy(stack=False),
                move=info["move"],
                ply_left=self.MAX_TREE_DEPTH - 1,
                alpha=-self.INF,
                beta=self.INF,
            )
            if tree is not None:
                trajectories.append({
                    "uci": info["move"].uci(),
                    "score": info["score"],
                    "tree": tree,
                })
        return trajectories

    def visualize_tree(self, tree: VariationNode) -> str:
        """Utility for quick CLI inspection."""
        return tree.visualize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ----------   Stockfish wrappers   ---------------------------------

    def _analyze(self, board: chess.Board, *, depth: int, multipv: int, think_time: float) -> List[Dict[str, Any]]:
        """Run Stockfish and structure its output (POV = *self._root_color*)."""
        limit = (
            chess.engine.Limit(time=think_time)
            if think_time > 0
            else chess.engine.Limit(depth=depth)
        )
        # Ensure _root_color is set before calling _structure_analysis
        if self._root_color is None:
             raise ValueError("Cannot analyze without _root_color being set.")

        raw = self._engine.analyse(board, limit=limit, multipv=multipv)
        return self._structure_analysis(raw, self._root_color)

    @classmethod
    def _structure_analysis(cls, raw: List[Dict[str, Any]], perspective: chess.Color) -> List[Dict[str, Any]]:
        """Convert engine JSON blobs → sorted move dictionaries."""
        moves: List[Dict[str, Any]] = []
        for entry in raw:
            score_obj = entry["score"].pov(perspective)
            if score_obj.is_mate():
                score = cls.MATE_SCORE * (1 if score_obj.mate() > 0 else -1)
                mate_in = score_obj.mate()
            else:
                score = score_obj.score(mate_score=cls.MATE_SCORE) or 0
                mate_in = None

            move = entry.get("pv", [entry.get("move")])[0]
            if move is None:
                continue

            moves.append({
                "move": move,
                "score": score,
                "is_mate": score_obj.is_mate(),
                "mate_in": mate_in,
            })

        return sorted(moves, key=lambda d: d["score"], reverse=True)

    # ----------   Leaf evaluation   -----------------------------------

    def _leaf_score(self, board: chess.Board) -> int:
        """Cheap fallback evaluation when we hit node/ply limits."""
        # Ensure _root_color is set before calling _analyze
        if self._root_color is None:
             raise ValueError("Cannot get leaf score without _root_color being set.")
        quick = self._analyze(board, depth=4, multipv=1, think_time=0.05)
        return quick[0]["score"] if quick else 0

    # ----------   Recursive tree construction   -----------------------

    def _build_tree(
        self,
        board: chess.Board,
        *,
        move: chess.Move,
        ply_left: int,
        alpha: int,
        beta: int,
    ) -> Optional[VariationNode]:
        """Depth‑limited α‑β with Stockfish leaf evaluations."""
        board.push(move)
        self._nodes_created += 1

        if board.is_game_over(claim_draw=True):
            is_checkmate = board.is_checkmate()
            score = self._terminal_score(board)
            mate_in = None
            if is_checkmate:
                 # Mate in 0 from opponent's perspective = mate in 1 for current player
                 mate_in = 1 if score == self.MATE_SCORE else -1
            node = VariationNode(move, score, score, is_checkmate, mate_in)
            board.pop()
            return node

        if ply_left <= 0 or self._nodes_created >= self.MAX_TREE_NODES:
            score = self._leaf_score(board)
            node = VariationNode(move, score, score) # Minimax = static eval at leaf
            board.pop()
            return node

        lines = self._analyze(board, **self._root_cfg)
        if not lines: # Should be caught by is_game_over, but defensive check
            score = self._leaf_score(board)
            node = VariationNode(move, score, score)
            board.pop()
            return node

        # Use score from analysis after the move for the node's static score
        current_score = lines[0]["score"]
        current_is_mate = lines[0]["is_mate"]
        current_mate_in = lines[0]["mate_in"]


        maximizing = board.turn == self._root_color
        ordered = lines if maximizing else list(reversed(lines))
        best_score = ordered[0]["score"]

        candidates = [ordered[0]] + [
            l for l in ordered[1:]
            if abs(l["score"] - best_score) <= self.CLEAR_BEST_THRESH
        ][: self.MAX_BRANCHING - 1]

        best_val = -self.INF if maximizing else self.INF
        children: List[VariationNode] = []
        best_child: Optional[VariationNode] = None

        a, b = alpha, beta
        for info in candidates:
            child = self._build_tree(
                board.copy(stack=False), # Use copy for parallel exploration if needed later
                move=info["move"],
                ply_left=ply_left - 1,
                alpha=a,
                beta=b,
            )
            if child is None:
                continue
            children.append(child)

            if maximizing:
                if child.minimax > best_val:
                    best_val = child.minimax
                    best_child = child
                a = max(a, best_val)
            else: # Minimizing
                if child.minimax < best_val:
                    best_val = child.minimax
                    best_child = child
                b = min(b, best_val)

            if b <= a:
                break

            if self._nodes_created >= self.MAX_TREE_NODES:
                break

        # If all children pruned or no legal moves, use static eval
        if not children or best_child is None:
            best_val = current_score # Use static eval if no children explored
            node_is_mate = current_is_mate
            node_mate_in = current_mate_in
        elif abs(best_val) == self.MATE_SCORE:
             node_is_mate = True
             # Propagate mate length from the best child
             if best_child.is_mate and best_child.mate_in is not None:
                 # Increment depth, preserve sign (positive = root wins, neg = opponent wins)
                 node_mate_in = (abs(best_child.mate_in) + 1) * (1 if best_val > 0 else -1)
             else:
                 # Mate found at this depth (best child was terminal/leaf mate)
                 node_mate_in = 1 if best_val > 0 else -1
        else: # Not a mate
             node_is_mate = False
             node_mate_in = None


        node = VariationNode(
            move,
            score=current_score, # Static score *after* this move
            minimax=best_val,   # Minimax score below this node
            is_mate=node_is_mate,
            mate_in=node_mate_in,
            children=sorted(children, key=lambda c: c.minimax, reverse=maximizing), # Sort children by their minimax value
        )
        board.pop()
        return node

    # ----------   Terminal evaluation helpers   -----------------------

    def _terminal_score(self, board: chess.Board) -> int:
        """Exact score for positions with no legal moves (mate / draw)."""
        # Ensure _root_color is set before calling _leaf_score
        if self._root_color is None:
             raise ValueError("Cannot get terminal score without _root_color being set.")

        if board.is_checkmate():
            # If it's opponent's turn after the move, root delivered mate
            return self.MATE_SCORE if board.turn != self._root_color else -self.MATE_SCORE
        # Check other draw conditions
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.is_repetition():
            return 0
        # Should not happen if is_game_over(claim_draw=True) is checked first, but fallback
        return self._leaf_score(board)