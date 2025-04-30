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
    opponenet_prune_branch_phrases,
    board_valuation_excellent_absolute,
    board_valuation_excellent_delta,
    board_valuation_good_absolute,
    board_valuation_good_delta,
    board_valuation_poor_absolute,
    board_valuation_poor_delta,
    board_valuation_blunder_absolute,
    board_valuation_blunder_delta
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
    INF = 10_000_000              # Sentinel for minimax initialisation
    MATE_CP   = 10_000            # Stockfish convention
    
    ROOT_WRITE_OFF_CP   = 100     # "bad strategy" threshold at root (cp)
    BRANCH_WRITE_OFF_CP = 60      # same idea for sub‑branches   (cp)

    # Determinants for move value
    GOOD_MOVE_CP = 50
    BAD_MOVE_CP = 50
    EXCELLENT_MOVE_CP = 100
    BLUNDER_CP = 100

    # Determinants for board value
    EXCELLENT_BOARD = 300
    GOOD_BOARD = 150
    BAD_BOARD = -150
    BLUNDER_BOARD = -300

    # RL-theory specific tuners
    NARRATE_BOARD_VALUE = True
    NARRATE_BOARD_VALUE_DELTA = True
    NARRATE_MOVE_VALUE = True
    SHOW_MOVE_VALUE = True
    SHOW_BOARD_VALUE = True
    
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

        # Need this to avoid confusion in the narration when jumping multiple depths
        self.previous_narration_depth = 0

    # ------------------------------------------------------------------ #
    #                         PUBLIC ENTRY POINT                         #
    # ------------------------------------------------------------------ #
    def generate_explanations(self) -> List[Dict[str, Any]]:
        """Generates explanations for each root entry and returns the updated list."""
        explanations = []
        depth_values = [self.initial_score]
        board = self.initial_board.copy()

        # Generate explanation for each entry
        for entry in self.root_entries:
            node = entry.get('tree')
            self.previous_narration_depth = 0   # Reset since back at root
            narrations, _ = self._generate_recursive_explanation(board, node, depth_values)
            # entry['explanation'] = self._sentence_casing(narrations)
            entry['explanation'] = narrations
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
        ) -> Tuple[List[str], bool]:
        """
        Primary function to recursively generate programmatic explanations using DFS technique.
        Returns:
            explanation_parts: List of explanations in natural language.
            is_writeoff: If true, means this node was written off (note that if this node's child is written off we don't propagate written off up)
        """
        explanation_parts: List[str] = []
        our_move = board.turn == self.root_color
        is_root = node.parent is None
        children_considered = 0

        # For all moves we will want to narrate
        if is_root:
            narration, is_writeoff = self._narrate_root(board, node)
        else:
            narration, is_writeoff = self._narrate_branch(board, node, depth_values, our_move)
        
        explanation_parts.extend(narration)   # Add to our list of explanation parts

        # Base case 1: Written off
        if is_writeoff:
            return explanation_parts, True
        
        # Base case 2: Leaf node
        if len(node.children) == 0:
            explanation_parts.append(self._narrate_board_value(node))
            return explanation_parts, False
        
        # Recursive case
        board.push(node.move)   # Update board for this move
        depth_values.append(node.score)
        for child in node.children:
            narrations, is_writeoff = self._generate_recursive_explanation(
                board=board,
                node=child,
                depth_values=depth_values
            )
            children_considered += 0 if is_writeoff else 1
            explanation_parts.extend(narrations)
        
        _ = depth_values.pop()
        _ = board.pop()

        # If multiple children considered, need to generate final 'best move' narration
        if children_considered > 1:
            explanation_parts.extend(
                self._narrate_best_move(board, node.children, our_move)
            )
        
        return explanation_parts, False

    # ................................................................. #    
    def _narrate_root(self, board: chess.Board, root: VariationNode) -> Tuple[List[str], bool]:
        """Narrate the root node and its children."""
        root_narration = []
        # First narrate the first move.
        move_description, _ = self._describe_move(board, root)
        root_narration.extend([
            random.choice(root_consider_phrases).format(
            move_description = move_description)
        ])

        # Now check if the root is a write-off
        if root.minimax < self.best_root_value - self.ROOT_WRITE_OFF_CP:
            root_narration.append(random.choice(write_off_root_phrases))
            return root_narration, True
        
        return root_narration, False
    

    def _narrate_branch(self, board, node, depth_values, our_move) -> Tuple[List[str, Any], bool]:
        """ Helper function to generate text for a branch node. """
        branch_text = []
        first_child = node == node.parent.children[0]

        prefix = self._get_depth_prefix(self.previous_narration_depth, node.depth, our_move)

        move_description, value_narration = self._describe_move(board, node, depth_values)
        if our_move:
            if first_child:
                branch_text.extend([
                    prefix + random.choice(our_move_first_child_phrases).format(move_description=move_description),
                    value_narration
                ])
            else:
                branch_text.extend([
                    prefix + random.choice(our_move_sibling_phrases).format(move_description=move_description),
                    value_narration
                ])
        else:
            if first_child:
                branch_text.extend([
                    prefix + random.choice(opponent_move_first_child_phrases).format(move_description=move_description),
                    value_narration
                ])
            else:
                branch_text.extend([
                    prefix + random.choice(opponent_move_sibling_phrases).format(move_description=move_description),
                    value_narration
                ])
        
        prune_text, is_writeoff = self._consider_branch(
            node=node, 
            our_move=our_move
        )
        branch_text.append(prune_text)

        return branch_text, is_writeoff


    def _consider_branch(self, node, our_move) -> Tuple[List[str], bool]:
        """ Checks if we should consider this branch or if we should prune. """
        if our_move:
            if node.minimax < node.parent.minimax - self.BRANCH_WRITE_OFF_CP:
                prune_text = random.choice(us_prune_branch_phrases)
                return prune_text, True
        else:
            if node.minimax > node.parent.minimax + self.BRANCH_WRITE_OFF_CP:
                prune_text = random.choice(opponenet_prune_branch_phrases)    
                return prune_text, True

        # If this is returned, we'll continue exploring that branch
        return None, False


    def _narrate_best_move(self, board, children, our_move) -> List[str]:
        """ Given a list of children picks the best one and returns in natural language. """
        best_child = None
        optimal_minimax = -self.INF if our_move else self.INF
        best_move_narration = ""

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
        move_description, _ = self._describe_move(board, best_child)
        if our_move:
            best_move_narration = random.choice(us_best_move_phrases).format(
                move_description=move_description)
        else:
            best_move_narration = random.choice(opponent_best_move_phrases).format(
                move_description=move_description)

        return [best_move_narration]


    def _describe_move(
            self, 
            board: chess.Board, 
            node: VariationNode, 
            depth_values: List[int] = None,
        ) -> Tuple[str, bool]:
        """ Function to, given a move, describe the move (and optionally narrate move value). """
        color = "white" if board.turn == chess.WHITE else "black"
        self.previous_narration_depth = node.depth   # Set for current node

        # Castling
        if board.is_castling(node.move):
            side = "kingside" if chess.square_file(node.move.to_square) == 6 else "queenside"
            return f"{color} castles {side} ({node.move.uci()})", None

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
        
        move_description = f"{color} {piece_name} {action} ({node.move.uci()})"

        # Narrate move value (if hyperparam set)
        value_narration = None
        if self.NARRATE_MOVE_VALUE and depth_values and len(depth_values) > 1:
            move_value = f"[{node.score-depth_values[-2]}]" if self.SHOW_MOVE_VALUE else ""
             # Case 1: Excellent move
            if node.score > (depth_values[-2] + self.EXCELLENT_MOVE_CP) and node.delta_score > self.EXCELLENT_MOVE_CP:
                value_narration = random.choice(excellent_move_phrases).format(move_value=move_value)
            # Case 2: Good move
            elif node.score > (depth_values[-2] + self.GOOD_MOVE_CP) and node.delta_score > self.GOOD_MOVE_CP:
                value_narration = random.choice(good_move_phrases).format(move_value=move_value)
            # Case 3: Blunder
            elif node.score < (depth_values[-2] - self.BLUNDER_CP) and node.delta_score < self.BLUNDER_CP:
                value_narration = random.choice(blunder_phrases).format(move_value=move_value)
            # Case 4: Bad move
            elif node.score < (depth_values[-2] - self.BAD_MOVE_CP) and node.delta_score > self.BAD_MOVE_CP:
                value_narration = random.choice(bad_move_phrases).format(move_value=move_value)
            # Case 5: Nothing to report, doesn't materially sway board
            else:
                pass
            return move_description, value_narration

        return move_description, value_narration


    def _narrate_board_value(self, node: VariationNode) -> List[str]:
        """
        Generates a statement about the value of a board (either on objective basis -- pure value function)
        or on a delta basis (compared vs. initial board score).
        Returns a list containing the narration string, or an empty list if no narration is generated.
        """
        if not self.NARRATE_BOARD_VALUE:
            return None

        delta_score = node.score - self.initial_score
        board_value = (f"[{delta_score}]" if delta_score else f"[{node.score}]") if self.SHOW_BOARD_VALUE else ""
        narration = None

        # Delta comparison
        if self.NARRATE_BOARD_VALUE_DELTA:
            if delta_score > self.EXCELLENT_BOARD:
                narration = random.choice(board_valuation_excellent_delta).format(board_value=board_value)
            elif delta_score > self.GOOD_BOARD:
                narration = random.choice(board_valuation_good_delta).format(board_value=board_value)
            elif delta_score < self.BLUNDER_BOARD:
                narration = random.choice(board_valuation_blunder_delta).format(board_value=board_value)
            elif delta_score < self.BAD_BOARD:
                narration = random.choice(board_valuation_poor_delta).format(board_value=board_value)
        # Absolute comparison
        else:
            if node.score > self.EXCELLENT_BOARD:
                narration = random.choice(board_valuation_excellent_absolute).format(board_value=board_value)
            elif node.score > self.GOOD_BOARD:
                narration = random.choice(board_valuation_good_absolute).format(board_value=board_value)
            elif node.score < self.BLUNDER_BOARD:
                narration = random.choice(board_valuation_blunder_absolute).format(board_value=board_value)
            elif node.score < self.BAD_BOARD:
                narration = random.choice(board_valuation_poor_absolute).format(board_value=board_value)

        return narration


    @staticmethod
    def _get_depth_prefix(prev_narration_depth, node_depth, our_move):
        depth_jump_size = (prev_narration_depth - node_depth)//2

        # Nominal - no need to prefix about which move we're returning to
        if depth_jump_size <= 0:
            return ""
        
        returning_to_move = node_depth // 2
        phrases = ['first', 'second', 'third'] # Won't take our tree deeper than this likely

        if our_move:
            return f"Returning to our {phrases[returning_to_move]} move, "
        else:
            return f"Returning to their {phrases[returning_to_move]} move, "


    @staticmethod
    def _sentence_casing(narrations: List[str, Any]) -> str:
        """Converts a List of string/None with multiple sentences into sentence case."""
        filtered_narrations = [narration for narration in narrations if narration is not None]    
        text = " ".join(filtered_narrations)
        
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