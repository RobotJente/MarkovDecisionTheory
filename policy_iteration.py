from utils import *
import numpy as np

# Parameters
discount = 0.95

# Made this string to avoid possible errors with integer multiplication (these should function as tokens, not integers)
states = ("Site 1", "Site 2", "Site 3", "Site 4")
trailer_states =  ("T Site 1", "T Site 2", "T Site 3", "T Site 4")
actions = ("Move to site 1", "Move to site 2", "Move to site 3", "Move to site 4")
transitions = np.zeros(shape=(4,4))
transitions[0] = [0.1, 0.3, 0.3, 0.3]
transitions[1] = [0, 0.5, 0.5, 0]
transitions[2] = [0, 0, 0.8, 0.2]
transitions[3] = [0.4, 0, 0, 0.6]

worker_locations = [0, 1, 2, 3]


# encode a way to find the result of an action (in this case, the actions bring the trailer from site i to site j
action_results = {}
for i in range(len(actions)):
    action_results[actions[i]] = states[i]


pi_0 = (actions[1], actions[2], actions[1], actions[2])

# evaluate the policy pi by calculating the state value function V
converged = False
pi = pi_0
while not converged:
    # initialize the trailer location
    current_trailer_loc = states[0]

    # reward vec comprises of only the short term rewards
    reward_vec = []
    for action, current_worker_loc in zip(pi, worker_locations):
        reward_vec.append(calc_total_expected_reward(current_worker_loc, current_trailer_loc, action, transitions, action_results, states))
        current_trailer_loc = action_results[action]
    print(reward_vec)

    # Policy evaluation:
    vn = np.dot(np.linalg.inv(np.eye(4) - 0.95*transitions), reward_vec)

    # Policy improvement:
    


    converged = True

