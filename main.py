from utils import *
import numpy as np
from value_iteration_rework import perform_val_it

# build the transition matrix
# transitions = np.zeros(shape=(2,2))
# transitions[0] = [0.99, 0.01]
# transitions[1] = [0.99, 0.01]
transitions = np.zeros(shape=(4,4))
transitions[0] = [0.1, 0.3, 0.3, 0.3]
transitions[1] = [0, 0.5, 0.5, 0]
transitions[2] = [0, 0, 0.8, 0.2]
transitions[3] = [0.4, 0, 0, 0.6]
dim_states = transitions.shape[0]

# initialization
epsilon = 1e-9
discount = 0.95

nodes = [i+1 for i in range(dim_states)]
actions = ["Move to site " + str(i+1) for i in range(dim_states)]

# encode a way to find the result of an action (in this case, the actions bring the trailer from site i to site j
action_results = {}
for i in range(len(actions)):
    action_results[actions[i]] = i + 1
perform_val_it(transitions, actions, nodes, discount, epsilon, action_results)


#
# # Parameters
# discount = 0.95
#
# # Made this string to avoid possible errors with integer multiplication (these should function as tokens, not integers)
# nodes = [1,2,3,4]
# actions = ("Move to site 1", "Move to site 2", "Move to site 3", "Move to site 4")
# #transitions = build_transition_matrix(nodes)
# transitions = np.zeros(shape=(4,4))
# transitions[0] = [0.1, 0.3, 0.3, 0.3]
# transitions[1] = [0, 0.5, 0.5, 0]
# transitions[2] = [0, 0, 0.8, 0.2]
# transitions[3] = [0.4, 0, 0, 0.6]
# # encode a way to find the result of an action (in this case, the actions bring the trailer from site i to site j
# action_results = {}
# for i in range(len(actions)):
#     action_results[actions[i]] = nodes[i]
#
#
# x = calc_cost_use_trailer(nodes[0], nodes[2], nodes)
# y = calc_cost_move_trailer(nodes[0], nodes[1])
# z = calc_immediate_expected_reward(nodes[0], nodes[0], actions[0], transitions, action_results, nodes)

############################################################

