root_consider_phrases = [
    "Ok let's consider starting with {move_description}.",
    "Hm, what if we played {move_description}?",
    "Let's rollout what would happen starting with {move_description}.",
    "Let's think through {move_description}.",
    "What if we consider {move_description}.",
    "We could try playing {move_description}.",
    "We could play {move_description}.",
    "Ok let's analyze the line starting with {move_description}."
    "What if we start with {move_description}?"
]

write_off_root_phrases = [
    "Hold that, this seems like a bad direction.",
    "No, not it."
    "No, this actually isn't the best line to consider.",
    "No, this doesn't seem like a good idea.",
    "Nevermind - let's consider something else.",
    "Bad idea. Not worth exploring.",
    "Eh this doesn't seem promising at all.",
    "Actually, not a good idea.",
    "Nope, this isn't a good move."
]

excellent_move_phrases = [
    "That looks like a really strong move.",
    "Hm this seems like a very move for me.",
    "Looks like it could be an excellent move.",
    "May be a brilliant move!",
    "I think that's a super strong move."
]

good_move_phrases = [
    "Looks to be a good move.", 
    "Seems like a strong move.",
    "I think this is a positive line to consider.",
    "This feels like a good direction.",
    "Just thinking out loud, seems like a good move.",
    "Good move - I like this."
]

bad_move_phrases = [
    "Looks like a poor line.",
    "Doesn't seem like a positive direction.",
    "Not a great move.",
    "This feels like a suboptimal direction.",
    "Seems like a bleh move."
]

blunder_phrases = [
    "Bleh, may be a blunder.",
    "Nope this seems like a very bad direction.",
    "That seems lke a very bad move.",
    "Hm, seems like a big mistake.",
    "Oh no we'd definitely be in trouble here."
]

our_move_first_child_phrases = [
    "We could then respond with {move_description}.",
    "Ok then we could play {move_description}."
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
    "Ok considering their moves, they may play {move_description}.",
    "They might hit back with {move_description}."
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
    "They would probably choose {move_description} of all the options."
    "I expect they would choose {move_description}.",
    "They would likely choose {move_description}.",
    "I think they would play {move_description}.",
    "Of these they would likely move {move_description}.",
]

us_prune_branch_phrases = [
    "We might think to play {move_description}, but that doesn't seem like a good idea.",
    "What about playing {move_description}...no that doesn't seem smart.",
    "We could consider {move_description}. On second thought that isn't a great line.",
    "We may think to play {move_description}, but it seems like a bad direction."
]

opponenet_prune_branch_phrases = [
    "They might consider {move_description}...but actually that would be pretty suboptimal for them to play.",
    "They could move {move_description}. However, that seems like a silly move for them.",
]