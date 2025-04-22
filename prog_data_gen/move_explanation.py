# move_explanation.py
from __future__ import annotations

import random
from typing import List, Dict, Any, Optional

import chess

from phrase_banks import (
    opponent_play_phrases,
    we_play_phrases,
    opponent_reply_intros,
    we_start_phrases,
    good_move_phrases,
    bad_move_phrases,
    leaf_verdict_phrases,
    balanced_verdict,
    positive_verdict,
    negative_verdict,
    multiple_move_phrases,
    write_off_phrases,
    best_move_superior_phrases,
    best_move_close_phrases
)


class VariationNode:  # minimal stub for type checkers / IDEs
    move: chess.Move
    score: int
    minimax: int
    children: List["VariationNode"]


class MoveExplanation:
    """
    Build an uncertain, human‑sounding commentary over a *list* of analysis
    entries (dictionaries containing VariationNode trees) produced by ChessExplainer.

    Key ideas
    ---------
    • Works from our side's point of view (initial_board.turn).
    • Iterates through each provided analysis entry.
    • For each entry with a valid tree, generates a narrative explanation.
    • First picks the best root move by minimax across all valid trees.
    • "Writes off" clearly inferior lines using two tunable cut‑offs:
         ROOT_WRITE_OFF_CP   – compared to the best root (within its own explanation)
         BRANCH_WRITE_OFF_CP – compared to the best sibling at that depth
    • Recurses until leaf nodes or until a branch falls under a write‑off.
    • Never shows raw centipawn numbers, only natural‑language verdicts.
    • Stores the generated explanation string back into the input dictionary.
    • Returns the list of dictionaries, now including explanations.
    """

    # ------------------------------------------------------------------ #
    #                        TUNABLE HYPERPARAMETERS                      #
    # ------------------------------------------------------------------ #
    ROOT_WRITE_OFF_CP   = 150     # "bad strategy" threshold at root (cp)
    BRANCH_WRITE_OFF_CP = 100     # same idea for sub‑branches   (cp)
    POS_DELTA = 50               # comment if eval jumps this much for us
    NEG_DELTA = -50              # comment if eval jumps this much for them
    MATE_CP   = 10_000           # Stockfish convention

    PIECE_NAMES = {
        chess.PAWN:   "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK:   "rook",
        chess.QUEEN:  "queen",
        chess.KING:   "king",
    }

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        initial_board: chess.Board,
        root_entries: List[Dict[str, Any]],
        initial_score: int
    ):
        self.initial_board = initial_board.copy()
        self.root_entries = root_entries
        self.initial_score = initial_score
        self._our_colour = self.initial_board.turn

        # Extract valid VariationNode roots from the dictionaries
        self.roots: List[VariationNode] = []
        for entry in self.root_entries:
            tree = entry.get('tree')
            if isinstance(tree, VariationNode):
                self.roots.append(tree)

        # Determine the best root move by minimax from OUR point of view IF roots exist
        self.best_root: Optional[VariationNode] = None
        self.best_root_value: Optional[int] = None
        if self.roots:
            self.best_root = (
                max(self.roots, key=lambda n: n.minimax)
                if self._our_colour == chess.WHITE
                else min(self.roots, key=lambda n: n.minimax)
            )
            self.best_root_value = self.best_root.minimax

    # ------------------------------------------------------------------ #
    #                         PUBLIC ENTRY POINT                          #
    # ------------------------------------------------------------------ #
    def generate_explanations(self) -> List[Dict[str, Any]]:
        """Generates explanations for each root entry and returns the updated list."""

        # Generate explanation for each entry
        for entry in self.root_entries:
            root_node = entry.get('tree')

            if isinstance(root_node, VariationNode) and self.best_root_value is not None:
                explanation_parts: List[str] = []

                # Intro if multiple plausible roots (added only to the first explanation generated)
                if len(self.roots) > 1 and entry['tree'] == self.roots[0]:
                    # Find the second root for the intro phrase (if it exists)
                    second_root = next((r for r in self.roots if r != root_node), None)
                    if second_root:
                        d1 = self._describe_move(self.initial_board, root_node.move)
                        d2 = self._describe_move(self.initial_board, second_root.move)
                        explanation_parts.append(
                            random.choice(multiple_move_phrases).format(move1=d1, move2=d2)
                        )

                # Generate the core explanation for this specific root
                explanation_parts.append(self._generate_single_explanation(root_node))

                # Add summary/verdict if this is the best root and there were alternatives
                if root_node == self.best_root and len(self.roots) > 1:
                    summary = self._generate_root_summary()
                    if summary:
                         explanation_parts.append(summary)

                entry['explanation'] = " ".join(explanation_parts)

            elif isinstance(root_node, VariationNode) and self.best_root_value is None:
                # Handle case where root exists but best_root couldn't be determined (e.g., only one root)
                 explanation_parts = [self._generate_single_explanation(root_node)]
                 entry['explanation'] = " ".join(explanation_parts)
            else:
                # Handle cases where the 'tree' is missing or not a VariationNode
                entry['explanation'] = "Analysis tree not available or invalid."

        return self.root_entries

    # ------------------------------------------------------------------ #
    #                          PRIVATE HELPERS                           #
    # ------------------------------------------------------------------ #

    def _generate_single_explanation(self, root: VariationNode) -> str:
        """Generates the narrative explanation for a single root VariationNode tree."""
        parts: List[str] = []
        board_after = self.initial_board.copy()

        # Is this root so bad we ignore it entirely compared to the absolute best?
        # Note: self.best_root_value could be None if no valid roots were found initially
        if self.best_root_value is not None and self._is_writeoff(root.minimax, self.best_root_value, self.ROOT_WRITE_OFF_CP):
            parts.append(random.choice(write_off_phrases).format(move=self._describe_move(self.initial_board, root.move)))
            return " ".join(parts)

        # Play and narrate root move
        move_prefix = random.choice(we_start_phrases)
        root_desc = self._describe_move(board_after, root.move)
        parts.append(f"{move_prefix}{root_desc}.")
        board_after.push(root.move)

        # Recurse through that root's tree
        if root.children:
            parts.append(self._narrate_children(board_after, root.children))
        else:
             # Handle case where root has no children (e.g., immediate game over or depth 0 analysis?)
             # This might need a specific phrase or just use the leaf verdict logic?
             # Using leaf verdict for now, assuming score is meaningful.
             parts.append(self._leaf_verdict(board_after, root))

        return " ".join(parts)

    def _narrate_children(
        self,
        board_before: chess.Board,
        children: List[VariationNode],
    ) -> str:
        """
        Narrate a *set* of sibling nodes at the same depth, then recurse
        into each in turn until leaf or write‑off.
        """
        out: List[str] = []

        # If >1 reply, introduce choices once
        if len(children) > 1:
            best, second = self._rank_children(children)[:2]
            m1 = self._describe_move(board_before, best.move)
            m2 = self._describe_move(board_before, second.move)
            out.append(
                random.choice(opponent_reply_intros)
                + " "
                + random.choice(multiple_move_phrases).format(move1=m1, move2=m2)
            )

        # Visit replies – best first, rest shuffled
        for child in self._rank_children(children):
            if self._is_writeoff_branch(board_before, child, children):
                out.append(random.choice(write_off_phrases))
                continue

            prefix_bank = (
                opponent_play_phrases
                if board_before.turn != self._our_colour
                else we_play_phrases
            )
            prefix = random.choice(prefix_bank)
            desc   = self._describe_move(board_before, child.move)
            out.append(f"{prefix} {desc}.")
            board_next = board_before.copy()
            board_next.push(child.move)

            # Delta comment
            delta = child.score - self.initial_score
            if delta >= self.POS_DELTA and board_before.turn == self._our_colour:
                out.append(random.choice(good_move_phrases))
            if delta <= self.NEG_DELTA and board_before.turn != self._our_colour:
                out.append(random.choice(bad_move_phrases))

            # Recurse
            if child.children:
                out.append(self._narrate_children(board_next, child.children))
            else:
                # Leaf: choose verdict and best leaf decision
                out.append(self._leaf_verdict(board_next, child))

        return " ".join(out)

    # ................................................................. #
    def _leaf_verdict(self, board: chess.Board, leaf: VariationNode) -> str:
        """Pick the best child (if any) or evaluate the leaf itself."""
        verdict = random.choice(leaf_verdict_phrases).format(
             move_desc = self._describe_move(board, leaf.move)
        )
        eval_phrase = self._evaluation_phrase(leaf.score)
        return f"{verdict} {eval_phrase}"

    # ................................................................. #
    def _generate_root_summary(self) -> str | None:
        """Compare best root to the rest and craft a short wrap‑up."""
        if len(self.roots) <= 1 or self.best_root is None:
            return None

        # Find the second best root that is not the best root
        valid_alternatives = [n for n in self.roots if n is not self.best_root]
        if not valid_alternatives:
             return None

        # Ensure best_root_value is not None before calculating diff
        if self.best_root_value is None:
             return None

        second_best = sorted(
            valid_alternatives,
            key=lambda n: n.minimax,
            reverse=self._our_colour == chess.WHITE
        )[0]

        # Check if second_best itself exists (should unless list was empty)
        if second_best is None:
             return None

        diff = abs(self.best_root_value - second_best.minimax)
        best_desc = self._describe_move(self.initial_board, self.best_root.move)

        if diff > self.ROOT_WRITE_OFF_CP:
            return random.choice(best_move_superior_phrases).format(best=best_desc)
        else:
            return random.choice(best_move_close_phrases).format(best=best_desc)

    # ................................................................. #
    def _rank_children(self, nodes: List[VariationNode]) -> List[VariationNode]:
        """Return children ordered best‑to‑worst for *our* side."""
        return sorted(
            nodes,
            key=lambda n: n.minimax,
            reverse=self._our_colour == chess.WHITE
        )

    def _is_writeoff(self, value: int, reference: int, thr: int) -> bool:
        if self._our_colour == chess.WHITE:
            return value < reference - thr
        return value > reference + thr

    def _is_writeoff_branch(
        self,
        board: chess.Board,
        node: VariationNode,
        siblings: List[VariationNode],
    ) -> bool:
        best_sibling = (
            max(siblings, key=lambda n: n.minimax)
            if self._our_colour == chess.WHITE
            else min(siblings, key=lambda n: n.minimax)
        )
        return self._is_writeoff(node.minimax, best_sibling.minimax, self.BRANCH_WRITE_OFF_CP)

    def _describe_move(self, board: chess.Board, move: chess.Move) -> str:
        colour = "White" if board.turn == chess.WHITE else "Black"

        # Castling
        if board.is_castling(move):
            side = "kingside" if chess.square_file(move.to_square) == 6 else "queenside"
            return f"{colour} castles {side} ({move.uci()})"

        piece = board.piece_at(move.from_square)
        piece_name = self.PIECE_NAMES.get(piece.piece_type, "piece")
        dest = chess.square_name(move.to_square)

        if board.is_capture(move):
            captured = (
                board.piece_at(move.to_square)
                if not board.is_en_passant(move)
                else board.piece_at(chess.square(move.to_square % 8, move.from_square // 8))
            )
            cap_name = self.PIECE_NAMES.get(captured.piece_type, "piece") if captured else "piece"
            action = f"captures the {cap_name} on {dest}"
        else:
            action = f"moves to {dest}"

        if move.promotion:
            promo = self.PIECE_NAMES[move.promotion]
            action += f", promoting to {promo}"

        return f"{colour} {piece_name} {action} ({move.uci()})"

    @staticmethod
    def _evaluation_phrase(score: int) -> str:
        if abs(score) < 50:
            return random.choice(balanced_verdict)
        return random.choice(positive_verdict if score > 0 else negative_verdict)
