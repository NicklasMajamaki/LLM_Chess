import gym



GYM_EVALS = [
    "bestmove"
]

# __main__() or whatever
# Instantiate our gym
# Outer loop happens in the main

# Class level - 'Gym'
# When you init -- you need to specify which evals you want to use (default to GYM_EVALS)
# When you init your gym, you'll have like a list of evals -- where each eval is its own class
# Each eval class should have similar core function names

# Main gym functionality:
    # instantiate boundary: 
    # rl_train: 
    # Both take the model as an arg -- you call these in the main function above

# Eval class:
    # determine_boundary:
        # """"""
    # evaluate