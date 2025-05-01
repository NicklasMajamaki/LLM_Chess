root_consider_phrases = [
    "Ok let's consider starting with {move_description}.",
    "Hm, what if we played {move_description}?",
    "Let's rollout what would happen starting with {move_description}.",
    "Let's think through {move_description}.",
    "What if we consider {move_description}.",
    "We could try playing {move_description}.",
    "We could play {move_description}.",
    "Ok let's analyze the line starting with {move_description}.",
    "What if we start with {move_description}?",
]

write_off_root_phrases = [
    "Hold that, this seems like a bad direction.",
    "No, not it.",
    "No, this actually isn't the best line to consider.",
    "No, this doesn't seem like a good idea.",
    "Nevermind - let's consider something else.",
    "Bad idea. Not worth exploring.",
    "Eh this doesn't seem promising at all.",
    "Actually, not a good idea.",
    "Nope, this isn't a good move."
]

excellent_move_phrases = [
    "That looks like a really strong move{move_value}.",
    "Hm this seems like a very move for me{move_value}.",
    "Looks like it could be an excellent move{move_value}.",
    "May be a brilliant move{move_value}!",
    "I think that's a super strong move{move_value}."
]

good_move_phrases = [
    "Looks to be a good move{move_value}.", 
    "Seems like a strong move{move_value}.",
    "I think this is a positive line to consider{move_value}.",
    "This feels like a good direction{move_value}.",
    "Just thinking out loud, seems like a good move{move_value}.",
    "Good move - I like this{move_value}."
]

bad_move_phrases = [
    "Looks like a poor line{move_value}.",
    "Doesn't seem like a positive direction{move_value}.",
    "Not a great move{move_value}.",
    "This feels like a suboptimal direction{move_value}.",
    "Seems like a bleh move{move_value}."
]

blunder_phrases = [
    "Bleh, may be a blunder{move_value}.",
    "Nope this seems like a very bad direction{move_value}.",
    "That seems like a very bad move{move_value}.",
    "Hm, seems like a big mistake{move_value}.",
    "Oh no we'd definitely be in trouble here{move_value}."
]

our_move_first_child_phrases = [
    "We could then respond with {move_description}.",
    "Ok then we could play {move_description}.",
    "We could follow with {move_description}.",
    "What if we then played {move_description}?",
    "Then we might play {move_description}.",
    "We could then move {move_description}.",
    "Consider following with {move_description}."
]

our_move_sibling_phrases = [
    "Ok we could instead play {move_description}.",
    "Instead we could move {move_description}.",
    "Alternatively, we could do {move_description}.",
    "Another possible move for us could be {move_description}.",
    "Or we could try playing {move_description}.",
    "What if instead we did {move_description}?",
    "We could also play {move_description}.",
    "We could also try {move_description}.",
    "Another option would be {move_description}."
]

opponent_move_first_child_phrases = [
    "They might think about responding with {move_description}.",
    "They could consider then playing {move_description}.",
    "Then they could respond with {move_description}.",
    "From here they may play {move_description}.",
    "They might hit back with {move_description}.",
    "They could consider replying with {move_description}."
]

opponent_move_sibling_phrases = [
    "They could also answer with {move_description}.",
    "They might also consider {move_description}.",
    "As an alternative, they could play {move_description}.",
    "Or they could move {move_description}.",
    "Also they might think about {move_description}.",
    "They could also consider {move_description}."
]

us_best_move_phrases = [
    "Of all of these, we should play {move_description}.", 
    "Out of those options, the best would be {move_description}.", 
    "We would choose {move_description} out of all the options.", 
    "I think of these we would choose {move_description}.", 
    "The best of which would be {move_description}.", 
]

opponent_best_move_phrases = [
    "They would probably choose {move_description} of all the options.",
    "I expect they would choose {move_description}.",
    "They would likely choose {move_description}.",
    "I think they would play {move_description}.",
    "Of these they would likely move {move_description}.",
]

us_prune_branch_phrases = [
    "No this branch doesn't seem right.",
    "Nevermind -- let's consider alternatives.",
    "Actually that isn't what we should play.",
    "No, maybe let's consider another move."
]

# This should be called VERY rarely (if at all)
opponenet_prune_branch_phrases = [
    "Actually they probably won't play that.",
    "On second thought they wouldn't choose this branch.",
]

board_valuation_excellent_absolute = [
    "We're in a really strong position to win here{board_value}.",
    "I really like our board positioning{board_value}.",
    "Our board position is looking very favorable{board_value}.",
    "We definitely have the upper hand here{board_value}.",
    "This board is very favorable for us{board_value}.",
    "If we keep this up we should definintely win{board_value}.",
]

board_valuation_good_absolute = [
    "This seems like a decent position for us{board_value}.",
    "I think we're in a reasonably good spot here{board_value}.",
    "The board looks somewhat favorable for us{board_value}.",
    "Feels like we have a slight advantage currently{board_value}.",
    "This position seems fairly solid{board_value}.",
]

board_valuation_poor_absolute = [
    "This position doesn't look great for us{board_value}.",
    "I'm a bit concerned about our current board state{board_value}.",
    "Seems like we might be at a disadvantage here{board_value}.",
    "This board feels somewhat unfavorable{board_value}.",
    "We could be in a tricky spot{board_value}.",
]

board_valuation_blunder_absolute = [
    "Oh, this looks like a really bad position for us{board_value}.",
    "This board is serious trouble for us{board_value}.",
    "This board state seems very unfavorable{board_value}.",
    "We're definitely playing from behind here{board_value}.",
    "This position looks quite difficult to come back from{board_value}.",
]