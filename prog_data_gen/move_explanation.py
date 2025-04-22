from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

import chess

if TYPE_CHECKING:                      # minimal type stubs
    class VariationNode:               # (real class lives elsewhere)
        move: chess.Move
        score: int                     # static evaluation *after* the move
        minimax: int                   # best‑play evaluation below this node
        children: List["VariationNode"]

    # Optional helper class – not used directly here, but kept for context
    # from chess_explainer import ChessExplainer


class MoveExplanation:
    """
    Convert a Stockfish‑style variation tree into flowing prose.

    - All moves are rendered "White <piece> ... (UCI)" or "Black <piece> ... (UCI)".
    - We speak in *first person* (“we / our”) for the root side.
    - Opponent replies use randomly‑chosen synonyms to avoid formulaic text.
    - Branches are shuffled so the 'best' reply is **not** always listed first.
    - At leaf nodes we mention the numeric evaluation, and at the end we
      highlight the most promising continuation according to the engine.
    """

    PIECE_NAMES = {
        chess.PAWN:   "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK:   "rook",
        chess.QUEEN:  "queen",
        chess.KING:   "king",
    }

    # Phrase banks for variety -------------------------------------------
    _OPPONENT_PLAY_PHRASES = [
        "Opponent plays",
        "Opponent counters with",
        "They respond with",
        "They answer with",
        "They could play",
        "They could move",
        "They may play",
        "They may move",
    ]

    _WE_PLAY_PHRASES = [
        "We continue with",
        "We follow up with",
        "We play",
        "Our reply is",
        "We could play",
        "We could move"
    ]

    _OPPONENT_REPLY_INTROS = [
        "From here the opponent has a few sensible options.",
        None,
        None,
        None,
        None,
    ]

    _WE_START_PHRASES = [
        "We could start with ",
        "Let's consider ",
        "Let's think about playing ",
        "What if we play ",
        "We could play ",
        "We could consider playing ",
    ]

    # thresholds and phrase banks for delta commentary and leaf verdicts
    POSITIVE_DELTA_THRESHOLD = 50
    NEGATIVE_DELTA_THRESHOLD = 50
    ADDITIONAL_IMPROVEMENT_THRESHOLD = 75  # For commenting on further improvements
    ADDITIONAL_DECLINE_THRESHOLD = 75      # For commenting on further declines
    _GOOD_MOVE_PHRASES = [
        "Excellent move!",
        "Strong play!",
        "That's a powerful move.",
        "A decisive improvement!",
        "This looks winning.",
        "Great tactical shot!",
        "Critical improvement!",
        "That's the key move."
    ]
    _BAD_MOVE_PHRASES = [
        "That's problematic for us.",
        "A troubling development.",
        "This creates serious issues.",
        "A concerning move.",
        "That's a setback.",
        "We're in trouble here.",
        "That move hurts our position.",
        "This could be critical."
    ]
    _FURTHER_IMPROVEMENT_PHRASES = [
        "This gets even better!",
        "Our advantage grows further.",
        "Another excellent move!",
        "We're piling on the pressure.",
        "This increases our edge.",
        "The position keeps improving.",
        "We're building our advantage.",
        "This strengthens our position even more."
    ]
    _FURTHER_DECLINE_PHRASES = [
        "Things are getting worse.",
        "Our position deteriorates further.",
        "Another serious problem.",
        "This is looking increasingly difficult.",
        "We're facing mounting challenges.",
        "Our defensive task is getting harder.",
        "The pressure keeps building against us.",
        "The situation continues to worsen."
    ]
    _LEAF_VERDICT_PHRASES = [
        "Of all those choices, {pronoun} should play {move}.",
        "Among those options, {pronoun} might opt for {move}.",
        "Of the possibilities, {pronoun} would likely choose {move}."
    ]
    MATE_CP = 10000
    _MATE_IN_SIGHT_PHRASES = [
        "This suggests a mate is in sight.",
        "It looks like a checkmate is looming in that direction.",
        "We may have a line of sight to a mate."
    ]
    _MATE_AGAINST_PHRASES = [
        "This may lead to a mate in their favor.",
        "They likely have a forced mate on that line.",
        "It appears to give them a mating opportunity."
    ]
    # ---------------------------------------------------------------------
    def __init__(self, initial_board: chess.Board, variation_root: "VariationNode"):
        self.initial_board = initial_board.copy()
        self.root = variation_root
        self._our_colour = self.initial_board.turn           # True = White

    # ---------------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------------
    def generate_explanation(self) -> str:
        """
        Return a single block of text narrating *all* branches beneath
        `self.root` and finishing with a short "best line" summary.
        """
        board = self.initial_board.copy()
        explanation_parts: List[str] = []

        # ----------------------------------------------------------------
        # 1. Our first move (root node)
        # ----------------------------------------------------------------
        root_desc = self._describe_move(board, self.root.move)
        explanation_parts.append(f"{random.choice(self._WE_START_PHRASES)}{root_desc}.")
        board.push(self.root.move)

        # Delta commentary for our first move if it's especially strong
        root_delta = getattr(self.root, "delta_score", None)
        if root_delta is not None and root_delta >= self.POSITIVE_DELTA_THRESHOLD:
            explanation_parts.append(random.choice(self._GOOD_MOVE_PHRASES))

        # ----------------------------------------------------------------
        # 2. Opponent's candidate replies – branches shuffled
        # ----------------------------------------------------------------
        if self.root.children:
            replies = list(self.root.children)
            random.shuffle(replies)                       # avoid fixed order
            intro = random.choice(self._OPPONENT_REPLY_INTROS)
            explanation_parts.append(intro)

            for child in replies:
                explanation_parts.append(self._narrate_branch(board.copy(), child, depth=1))

        # ----------------------------------------------------------------
        # 3. Engine verdict – pick the path whose leaf minimax matches
        #    the root's minimax (i.e. best play for both sides)
        # ----------------------------------------------------------------
        best_path = self._find_best_path(self.root, maximizing=True)
        best_leaf = best_path[-1]
        best_score_cp = best_leaf.score
        
        # Describe the *first* reply in that best line (i.e. opponent's move)
        board_for_best = self.initial_board.copy()
        board_for_best.push(self.root.move)
        best_reply_desc = (
            self._describe_move(board_for_best, best_path[1].move)
            if len(best_path) >= 2 else "the resulting position"
        )

        # Define verdict phrases based on score
        _BEST_LINE_POSITIVE = [
            f"Overall, the continuation beginning with {best_reply_desc} gives us a clear advantage.",
            f"The line starting with {best_reply_desc} leaves us in a favorable position.",
            f"Playing through {best_reply_desc} puts us in a strong position.",
            f"The sequence beginning with {best_reply_desc} gives us good winning chances.",
            f"Following the line with {best_reply_desc} maintains our advantage."
        ]
        
        _BEST_LINE_NEGATIVE = [
            f"Overall, the continuation beginning with {best_reply_desc} leaves us at a disadvantage.",
            f"The line starting with {best_reply_desc} is challenging for our position.",
            f"Playing through {best_reply_desc} requires careful defensive play.",
            f"The sequence beginning with {best_reply_desc} favors our opponent.",
            f"Following the line with {best_reply_desc} means we'll need to find resources."
        ]
        
        _BEST_LINE_BALANCED = [
            f"Overall, the continuation beginning with {best_reply_desc} keeps the position balanced.",
            f"The line starting with {best_reply_desc} maintains approximate equality.",
            f"Playing through {best_reply_desc} leads to an even struggle.",
            f"The sequence beginning with {best_reply_desc} results in balanced chances.",
            f"Following the line with {best_reply_desc} keeps the game level."
        ]
        
        # Select appropriate verdict based on score
        if abs(best_score_cp) < 50:
            summary = random.choice(_BEST_LINE_BALANCED)
        elif best_score_cp > 0:
            summary = random.choice(_BEST_LINE_POSITIVE)
        else:
            summary = random.choice(_BEST_LINE_NEGATIVE)
            
        explanation_parts.append(summary)
        explanation_parts = [part for part in explanation_parts if part is not None]    
        return " ".join(explanation_parts)

    # ---------------------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------------------
    def _narrate_branch(
        self,
        board_before: chess.Board,
        node: "VariationNode",
        depth: int,
        max_score_seen: int = None,
        min_score_seen: int = None,
    ) -> str:
        """
        Recursively narrate one branch starting with `node` (whose move is
        about to be played on `board_before`).  `depth` controls indentation
        in the recursive phrasing, but here we only use it for random‑choice
        variety (deeper lines tend to get shorter wording).
        
        max_score_seen and min_score_seen track the maximum and minimum scores
        seen so far in this branch to determine if a move is worth commenting on.
        """
        branch_parts: List[str] = []

        # Initialize max/min scores if this is the first call
        if max_score_seen is None:
            max_score_seen = -float('inf')
        if min_score_seen is None:
            min_score_seen = float('inf')

        # ----------------------------------------------------------------
        # 1. Play the move, describe it
        # ----------------------------------------------------------------
        side_to_move = board_before.turn
        phrase_bank = (
            self._WE_PLAY_PHRASES if side_to_move == self._our_colour
            else self._OPPONENT_PLAY_PHRASES
        )
        prefix = random.choice(phrase_bank)

        move_desc = self._describe_move(board_before, node.move)
        branch_parts.append(f"{prefix} {move_desc}.")
        
        # Delta commentary with improved logic
        delta = getattr(node, "delta_score", None)
        current_score = getattr(node, "score", 0)
        
        if delta is not None:
            if side_to_move == self._our_colour:
                if delta >= self.POSITIVE_DELTA_THRESHOLD and current_score > max_score_seen:
                    # First significant improvement or additional significant improvement
                    if current_score > max_score_seen + self.ADDITIONAL_IMPROVEMENT_THRESHOLD:
                        branch_parts.append(random.choice(
                            self._FURTHER_IMPROVEMENT_PHRASES if max_score_seen > -float('inf') 
                            else self._GOOD_MOVE_PHRASES
                        ))
                    elif max_score_seen == -float('inf'):
                        branch_parts.append(random.choice(self._GOOD_MOVE_PHRASES))
                    # Update max score seen
                    max_score_seen = max(max_score_seen, current_score)
            elif side_to_move != self._our_colour:
                if delta <= -self.NEGATIVE_DELTA_THRESHOLD and current_score < min_score_seen:
                    # First significant decline or additional significant decline
                    if current_score < min_score_seen - self.ADDITIONAL_DECLINE_THRESHOLD:
                        branch_parts.append(random.choice(
                            self._FURTHER_DECLINE_PHRASES if min_score_seen < float('inf') 
                            else self._BAD_MOVE_PHRASES
                        ))
                    elif min_score_seen == float('inf'):
                        branch_parts.append(random.choice(self._BAD_MOVE_PHRASES))
                    # Update min score seen
                    min_score_seen = min(min_score_seen, current_score)

        board_after = board_before.copy()
        board_after.push(node.move)

        # ----------------------------------------------------------------
        # 2. Recurse over children (shuffled)
        # ----------------------------------------------------------------
        if node.children:
            children = list(node.children)
            random.shuffle(children)

            for child in children:
                branch_parts.append(
                    self._narrate_branch(
                        board_after.copy(), 
                        child, 
                        depth + 1,
                        max_score_seen,
                        min_score_seen
                    )
                )
            # Verdict on leaf choices
            if len(children) > 1:
                # Determine best continuation
                if board_after.turn == self._our_colour:
                    best_child = max(children, key=lambda n: n.minimax)
                else:
                    best_child = min(children, key=lambda n: n.minimax)
                best_move_desc = self._describe_move(board_after, best_child.move)
                pronoun = "we" if board_after.turn == self._our_colour else "they"
                branch_parts.append(
                    random.choice(self._LEAF_VERDICT_PHRASES).format(pronoun=pronoun, move=best_move_desc)
                )
                # Mate sight commentary
                if best_child.score >= self.MATE_CP:
                    branch_parts.append(random.choice(self._MATE_IN_SIGHT_PHRASES))
                elif best_child.score <= -self.MATE_CP:
                    branch_parts.append(random.choice(self._MATE_AGAINST_PHRASES))
        else:
            # Leaf → give verdict
            branch_parts.append(self._evaluation_phrase(node.score))
        
        # Remove all 'None' from branch_parts
        branch_parts = [part for part in branch_parts if part is not None]
        return " ".join(branch_parts)

    # .....................................................................
    def _describe_move(self, board: chess.Board, move: chess.Move) -> str:
        """
        Return text of the form "White knight captures on e4 (d2e4)" or "Black knight captures on e4 (d2e4)".
        Castling and promotions get special wording.
        """
        colour = "White" if board.turn == chess.WHITE else "Black"

        # Castling --------------------------------------------------------
        if board.is_castling(move):
            side = "kingside" if chess.square_file(move.to_square) == 6 else "queenside"
            return f"{colour} castles {side} ({move.uci()})"

        piece = board.piece_at(move.from_square)
        piece_name = self.PIECE_NAMES.get(piece.piece_type, "piece")

        dest_sq = chess.square_name(move.to_square)

        # Capture? --------------------------------------------------------
        if board.is_capture(move):
            # Try to name captured piece
            captured_piece = (
                board.piece_at(move.to_square)
                if not board.is_en_passant(move)
                else board.piece_at(chess.square(move.to_square % 8, move.from_square // 8))
            )
            cap_name = (
                self.PIECE_NAMES.get(captured_piece.piece_type, "piece")
                if captured_piece else "piece"
            )
            action = f"captures the {cap_name} on {dest_sq}"
        else:
            action = f"moves to {dest_sq}"

        # Promotion? ------------------------------------------------------
        if move.promotion:
            promo_name = self.PIECE_NAMES[move.promotion]
            action += f", promoting to {promo_name}"

        return f"{colour} {piece_name} {action} ({move.uci()})"

    # .....................................................................

    _BALANCED_VERDICT = [
        "The position remains roughly balanced.",
        "Board seems like it remains even.",
        "Looks like the position is roughly even.",
        "Board looks to remain balanced.",
        "This position seems to be even.",
        "Neither side has a clear advantage here.",
        "The game remains relatively equal.",
        "Both sides have comparable chances.",
        "We're in a fairly balanced situation.",
        "The position is approximately equal."
    ]

    _POSITIVE_VERDICT = [
        "We hold a pleasant edge.",
        "We have a clear advantage.",
        "Our position is notably stronger.",
        "We've gained a tangible advantage.",
        "We're in a favorable position.",
        "Our pieces are working well together.",
        "We've secured a promising advantage.",
        "We have the upper hand in this position.",
        "Our position looks quite promising.",
        "We've obtained a comfortable advantage."
    ]

    _NEGATIVE_VERDICT = [
        "We're at a disadvantage.",
        "Our position is somewhat worse.",
        "We face some challenges ahead.",
        "The opponent has gained an edge.",
        "We'll need to play carefully from here.",
        "Our position has some weaknesses.",
        "We're on the defensive for now.",
        "The opponent has the upper hand.",
        "We're facing a difficult position.",
        "We need to find resources to equalize."
    ]

    @staticmethod
    def _evaluation_phrase(score_cp: int) -> str:
        """ Give a quick verdict on the position. """
        if abs(score_cp) < 50:
            return random.choice(MoveExplanation._BALANCED_VERDICT)
        if score_cp > 0:
            return random.choice(MoveExplanation._POSITIVE_VERDICT)
        return random.choice(MoveExplanation._NEGATIVE_VERDICT)

    # .....................................................................
    def _find_best_path(
        self,
        node: "VariationNode",
        maximizing: bool,
    ) -> List["VariationNode"]:
        """
        Follow the minimax numbers to recover *one* principal variation that
        achieves `node.minimax`.

        `maximizing` should be True when it's *our* turn at `node`, False
        when it's the opponent's.
        """
        path = [node]
        current = node
        maximize = maximizing

        while current.children:
            if maximize:
                best_child = max(current.children, key=lambda n: n.minimax)
            else:
                best_child = min(current.children, key=lambda n: n.minimax)

            path.append(best_child)
            current = best_child
            maximize = not maximize   # alternate sides

        return path