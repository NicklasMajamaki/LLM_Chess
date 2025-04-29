from __future__ import annotations

import re
import random
from typing import List, Dict, Any, Optional, Tuple

import chess

from variation_node import VariationNode


# ------------------------------------------------------------------
# Main API for turning tree to natural language
# ------------------------------------------------------------------
def generate_reasoning(
        initial_board: chess.Board,
        root_entries: List[Dict[str, Any]],
        initial_score: int
    ) -> List[Dict[str, Any]]:
        """ External-facing function to generate an explanation. Wraps around the MoveExplanation class """
        explainer = MoveExplanation(initial_board, root_entries, initial_score)
        return explainer.generate_explanations()


# ------------------------------------------------------------------
# Helper object for move explanations 
# ------------------------------------------------------------------
from phrase_banks import (
    root_consider_phrases,
    write_off_root_phrases,
    excellent_move_phrases,
    good_move_phrases,
    bad_move_phrases,
    blunder_phrases,
    our_move_first_child_phrases,
    our_move_sibling_phrases,
    opponent_move_first_child_phrases,
    opponent_move_sibling_phrases,
    us_best_move_phrases,
    opponent_best_move_phrases,
    us_prune_branch_phrases,
    opponenet_prune_branch_phrases
)


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
    GOOD_MOVE_CP = 50
    BAD_MOVE_CP = 50
    EXCELLENT_MOVE_CP = 100
    BLUNDER_CP = 100

    INF = 10_000_000              # Sentinel for minimax initialisation
    MATE_CP   = 10_000            # Stockfish convention

    # RL-theory specific tuners
    NARRATE_BOARD_VALUE = True
    NARRATE_MOVE_VALUE = True
    
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
        self.root_color = self.initial_board.turn

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
                max(self.roots, key=lambda n: n.minimax))
            self.best_root_value = self.best_root.minimax

    # ------------------------------------------------------------------ #
    #                         PUBLIC ENTRY POINT                         #
    # ------------------------------------------------------------------ #
    def generate_explanations(self) -> List[Dict[str, Any]]:
        """Generates explanations for each root entry and returns the updated list."""
        explanations = []

        # Generate explanation for each entry
        for entry in self.root_entries:
            root_node = entry.get('tree')

            if isinstance(root_node, VariationNode) and self.best_root_value is not None:
                explanation_parts: List[str] = []

                # First get the root & if this is a write-off
                branch_board = self.initial_board.copy()   # Just to be safe w/ mutation
                root_desc, is_writeoff = self._narrate_root(branch_board, root_node)
                explanation_parts.append(root_desc)
                if is_writeoff:
                    entry['explanation'] = " ".join(explanation_parts)
                    explanations.append(entry)
                    continue
                
                # If not a write-off move (i.e., worth considering) now branch out for logic
                depth_values = [self.initial_score]  # Will use this as reference for 'good' move or 'bad' move
                explanation_parts.extend(self._generate_recursive_explanation(branch_board, root_node, depth_values, is_root=True))
                entry['explanation'] = self._sentence_casing(" ".join(explanation_parts))
                explanations.append(entry)

        return explanations

    # ------------------------------------------------------------------ #
    #                          PRIVATE HELPERS                           #
    # ------------------------------------------------------------------ #
    def _generate_recursive_explanation(
            self, 
            board: chess.Board, 
            node: VariationNode,
            depth_values: List[int],
            is_root: bool,
            first_child: bool = False
        ) -> List[str]:
        """
        Primary function to recursively generate programmatic explanations.
        Returns:
            explanation_parts: List of explanations in natural language.
            bool: If true, means it was a considered (i.e., not written off) move
        """
        explanation_parts: List[str] = []
        our_move = not board.turn == self.root_color # Since we're looking at the move following 'node'

        # Start by narrating our move (if not the root)
        if not is_root:
            explanation_parts.append(self._narrate_branch(
                board=board,
                node=node,
                depth_values=depth_values,
                our_move=our_move,
                first_child=first_child
            ))
        
        # Base case: Leaf
        if not node.children:
            # TODO: include some 'value' of the board for end state (TD-lambda rollout for Value Iter)
            return explanation_parts, True

        # Recursively analyze children
        moves_considered = 0
        for i, child in enumerate(node.children):
            resp, alive = self._consider_branch(board, node, child, our_move)
            if not alive:   # Case where we prune this branch
                explanation_parts.append(resp)
                continue

            next_board = board.push(node.move)
            child_explanations, considered = self._generate_recursive_explanation(
                    board=next_board,
                    node=child,
                    depth_values=depth_values + [node.score],
                    is_root=False,
                    first_child=(i==0)
                )
            moves_considered += considered
            explanation_parts.extend(child_explanations)
            _ = board.pop()

        # If multiple moves analyzed narrate what the 'best' would be
        if moves_considered > 1:
            explanation_parts.append(
                self._narrate_best_move(board, node.children, our_move)
            )

        return explanation_parts, True

    # ................................................................. #    
    def _narrate_branch(self, board, node, depth_values, our_move, first_child) -> str:
        """ Helper function to generate text for a branch node. """
        branch_text = ""
        if our_move:
            if first_child:
                branch_text = random.choice(our_move_first_child_phrases).format(move_description=self._describe_move(board, node, depth_values))
            else:
                branch_text = random.choice(our_move_sibling_phrases).format(move_description=self._describe_move(board, node, depth_values))
        else:
            if first_child:
                branch_text = random.choice(opponent_move_first_child_phrases).format(move_description=self._describe_move(board, node, depth_values))
            else:
                branch_text = random.choice(opponent_move_sibling_phrases).format(move_description=self._describe_move(board, node, depth_values))
        
        return branch_text
    

    def _consider_branch(self, board, parent, child, our_move) -> Tuple[str, bool]:
        """ Checks if we should consider this branch or if we should prune. """
        if our_move:
            if child.minimax < parent.minimax - self.BRANCH_WRITE_OFF_CP:
                prune_text = random.choice(us_prune_branch_phrases).format(move_description=self._describe_move(board, child))
                return prune_text, False
        else:
            if child.minimax > parent.minimax + self.BRANCH_WRITE_OFF_CP:
                prune_text = random.choice(opponenet_prune_branch_phrases).format(move_description=self._describe_move(board, child))
                return prune_text, False

        # If this is returned, we'll continue exploring that branch
        return "", True


    def _narrate_best_move(self, board, children, our_move):
        """ Given a list of children picks the best one and returns in natural language. """
        best_child = None
        optimal_minimax = -self.INF if our_move else self.INF
        best_move_text = ""

        # Find the best child
        for child in children:
            if our_move:
                if child.minimax > optimal_minimax:
                    optimal_minimax = child.minimax
                    best_child = child
            else:
                if child.minimax < optimal_minimax:
                    optimal_minimax = child.minimax
                    best_child = child
        
        # Narrate move
        if our_move:
            best_move_text = random.choice(us_best_move_phrases).format(
                move_description=self._describe_move(board, best_child
            ))
        else:
            best_move_text = random.choice(opponent_best_move_phrases).format(
                move_description=self._describe_move(board, best_child
            ))

        return best_move_text

    
    def _narrate_root(self, board: chess.Board, root: VariationNode) -> Tuple[str, bool]:
        """Narrate the root node and its children."""
        # First narrate the first move.
        root_desc = random.choice(root_consider_phrases).format(
            move_description = self._describe_move(board, root, depth_values=None)
        )

        # Now check if the root is a write-off
        if root.minimax < self.best_root_value - self.ROOT_WRITE_OFF_CP:
            root_desc += " " + random.choice(write_off_root_phrases)
            return root_desc, True
        return root_desc, False

    @staticmethod
    def _sentence_casing(text: str) -> str:
        """Converts a string with multiple sentences into sentence case."""
        # Split text into sentences using regex to handle '.', '!', and '?'
        sentences = re.split(r'([.!?])', text)
        processed_sentences = []

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip().capitalize()
            punctuation = sentences[i + 1]
            processed_sentences.append(sentence + punctuation)

        # Handle any trailing text without punctuation
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            processed_sentences.append(sentences[-1].strip().capitalize())

        return " ".join(processed_sentences)

    def _describe_move(
            self, 
            board: chess.Board, 
            node: VariationNode, 
            depth_values: List[int] = None
        ) -> str:
        """ Function to, given a move, describe the move (and optionally narrate move value). """
        colour = "white" if board.turn == chess.WHITE else "black"

        # Castling
        if board.is_castling(node.move):
            side = "kingside" if chess.square_file(node.move.to_square) == 6 else "queenside"
            return f"{colour} castles {side} ({node.move.uci()})"

        piece = board.piece_at(node.move.from_square)
        piece_name = self.PIECE_NAMES.get(piece.piece_type, "piece")
        dest = chess.square_name(node.move.to_square)

        if board.is_capture(node.move):
            captured = (
                board.piece_at(node.move.to_square)
                if not board.is_en_passant(node.move)
                else board.piece_at(chess.square(node.move.to_square % 8, node.move.from_square // 8))
            )
            cap_name = self.PIECE_NAMES.get(captured.piece_type, "piece") if captured else "piece"
            action = f"captures the {cap_name} on {dest}"
        else:
            action = f"moves to {dest}"

        if node.move.promotion:
            promo = self.PIECE_NAMES[node.move.promotion]
            action += f", promoting to {promo}"
        
        # Clone the board and make the move to check for check/checkmate
        test_board = board.copy()
        test_board.push(node.move)
        
        if test_board.is_checkmate():
            action += " delivering checkmate"
        elif test_board.is_check():
            action += " putting the king in check"
        
        move_description = f"{colour} {piece_name} {action} ({node.move.uci()})."

        # Narrate move value (if hyperparam set)
        if self.NARRATE_MOVE_VALUE and depth_values and len(depth_values) > 1:
             # Case 1: Excellent move
            if node.score > (depth_values[-2] + self.EXCELLENT_MOVE_CP) and node.delta_score > self.EXCELLENT_MOVE_CP:
                move_description += random.choice(excellent_move_phrases)
            # Case 2: Good move
            elif node.score > (depth_values[-2] + self.GOOD_MOVE_CP) and node.delta_score > self.GOOD_MOVE_CP:
                move_description += random.choice(good_move_phrases)
            # Case 3: Blunder
            elif node.score < (depth_values[-2] - self.BLUNDER_CP) and node.delta_score < self.BLUNDER_CP:
                move_description += random.choice(blunder_phrases)
            # Case 4: Bad move
            elif node.score < (depth_values[-2] - self.BAD_MOVE_CP) and node.delta_score > self.BAD_MOVE_CP:
                move_description += random.choice(bad_move_phrases)
            # Case 5: Nothing to report, doesn't materially sway board
            else:
                pass

        return move_description