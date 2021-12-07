print("hello world")

discount = 0.95

# Made this string to avoid possible errors with integer multiplication (these should function as tokens, not integers)
states = ("Site 1", "Site 2", "Site 3", "Site 4")
actions = ("Move to site 1", "Move to site 2", "Move to site 3", "Move to site 4")


# return the cost for moving the trailer between locations
def calc_cost_move_trailer(current_loc, future_loc):
    pass

# return the cost for using the trailer if the workers are at location worker_loc, and the trailer is at location trailer_loc
def calc_cost_use_trailer(current_loc, future_loc):
    pass

# not sure what the best way is to represent this. We can also split it up per state and do the transitions that way.
# since we probably have to use dynamic programming to solve this, that might be a better way of encoding the transitions
def get_transition_matrix():
    pass