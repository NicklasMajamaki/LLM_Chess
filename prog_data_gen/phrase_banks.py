root_consider_phrases = [
    "We could consider {move_description}.",
    "Let's think about playing {move_description}.",
    "What if we do {move_description}.",
    "Let's think through {move_description}.",
    "Consider {move_description}.",
    "We could play {move_description}.",
    "Let's think about the line starting with {move_description}.",
    "Ok, consider {move_description}.",
]

write_off_root_phrases = [
    "This seems like a bad direction.",
    "This isn't the best line to consider.",
    "No, this doesn't seem like a good idea.",
    "Nope nevermind, let's consider something else.",
    "Bad idea, not worth exploring.",
    "This doesn't seem promising at all.",
    "Actually, not a good idea.",
    "Nevermind, this isn't a good move."
]

excellent_move_phrases = [
    "That looks like a really strong move.",
    "Seems to really tilt the board in my favor.",
    "Looks like it could be an excellent move.",
    "May be a brilliant move!",
    "I think that's a super strong move!"
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
    "Seems like a total blunder.",
    "Nope this seems like a very bad direction.",
    "Not this - I don't like this at all.",
    "Hm, seems like a big mistake.",
    "We'd definitely be in trouble here."
]

our_move_first_child_phrases = [
    "Let's first consider {move_description}.",
    "We could follow with {move_description}.",
    "What if we played {move_description}.",
    "We could play {move_description}.",
    "We could move {move_description}.",
    "What might we play? Let's start with {move_description}.",
    "Consider {move_description}."
]

our_move_sibling_phrases = [
    "We could also consider {move_description}.",
    "What if instead we did {move_description}.",
    "Instead we could play {move_description}.",
    "We could also play {move_description}.",
    "What about if we responded with {move_description}.",
    "Ok or we could move {move_description}.",
    "We could also try {move_description}.",
    "Or we could try playing {move_description}.",
    "Considering alternatives, {move_description}.",
    "Another option would be {move_description}."
]

opponent_move_first_child_phrases = [
    "They might think about responding with {move_description}.",
    "They may consider playing {move_description}.",
    "Let's think about their response, starting with {move_description}.",
    "Ok considering their moves, they may play {move_description}.",
    "From their side they may want to play {move_description}.",
    "They could consider playing {move_description}."
]

opponent_move_sibling_phrases = [
    "They could also answer with {move_description}.",
    "They might also consider {move_description}.",
    "As an alternative, they could play {move_description}.",
    "Or, they could move {move_description}.",
    "Also they might think about {move_description}.",
    "The opponent could also consider {move_description}."
]

us_best_move_phrases = [
    "Of all of these, we should play {move_description}.", 
    "Out of those options, the best would be {move_description}.", 
    "We would choose {move_description} out of all the options.", 
    "I think of these we would choose {move_description}.", 
    "The best of which would be {move_description}.", 
]

opponent_best_move_phrases = [
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








opponent_reply_intros = [
    "From here the opponent has a few sensible options.",
    None,
    None,
    None,
    None,
]


leaf_verdict_phrases = [
    "Of all those choices, {pronoun} should play {move}.",
    "Among those options, {pronoun} might opt for {move}.",
    "Of the possibilities, {pronoun} would likely choose {move}."
]

balanced_verdict = [
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

positive_verdict = [
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

negative_verdict = [
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

multiple_move_phrases = [
    "I could see myself playing {move1} or possibly {move2}.",
    "Both {move1} and {move2} come to mind here.",
    "Perhaps {move1} is sensible, though {move2} also looks tempting."
]

best_move_superior_phrases = [
    "Overall, {best} feels clearly superior to every alternative.",
    "In the end, {best} stands head and shoulders above the rest.",
    "Given the options, {best} is obviously the way to go."
]

best_move_close_phrases = [
    "After weighing everything, {best} edges out the other choices.",
    "It's close, but {best} appears a shade more promising.",
    "All things considered, I'd lean toward {best}."
]
