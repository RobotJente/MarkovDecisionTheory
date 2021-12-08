from utils import *


# Parameters
discount = 0.95

# Made this string to avoid possible errors with integer multiplication (these should function as tokens, not integers)
states = ("Site 1", "Site 2", "Site 3", "Site 4")
actions = ("Move to site 1", "Move to site 2", "Move to site 3", "Move to site 4")
transitions = build_transition_matrix(states)

# encode a way to find the result of an action (in this case, the actions bring the trailer from site i to site j
action_results = {}
for i in range(len(actions)):
    action_results[actions[i]] = states[i]


x = calc_cost_use_trailer(states[0], states[2], states)
y = calc_cost_move_trailer(states[0], states[1])
z = calc_total_expected_reward(states[0], actions[0], transitions, action_results, states)

print(z)