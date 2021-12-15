import numpy as np
from utils import calc_immediate_expected_reward
from main import action_results, transitions
import copy
# value iteration algorithm

# initialization
epsilon = 1e-5
discount = 0.95

# initialize v for each state
v = np.ones([16,1])
err = 1
pi = [None for i in range(len(v))] # here we will save the policies
nodes = ("Site 1", "Site 2", "Site 3", "Site 4")
actions = ("Move to site 1", "Move to site 2", "Move to site 3", "Move to site 4")


n = 0
while (err > epsilon * (1 - discount) / (2 * discount)):
    n += 1

    # track old value to determine convergence. use deepcopy since python normally tracks the reference, not the value
    v_old = copy.deepcopy(v)

    vector_index = 0
    for current_trailer_loc in nodes:
        for current_worker_loc in nodes:
            reward_vector = np.zeros([4, 1])

            # probability matrix for that action
            P = np.zeros([4, 16])

            # for the current state, find the reward of each action.
            for idx, a in enumerate(actions):
                reward_vector[idx] = calc_immediate_expected_reward(current_trailer_loc, current_worker_loc, a,
                                                                    transitions, action_results, nodes)
                matrix_column_index = 0
                for future_trailer_loc in nodes:
                    for future_worker_loc in nodes:
                        P[idx][matrix_column_index] = (action_results[a] == future_trailer_loc) * \
                                                      transitions[nodes.index(current_worker_loc)][
                                                          nodes.index(future_worker_loc)]
                        matrix_column_index += 1
            # For each state, choose v to be the result of the action with the highest reward
            v[vector_index] = np.max(reward_vector + discount * np.dot(P, v))

            # Keep track of the policy only for display results (not used in calculations)
            pi[vector_index] = actions[np.argmax(reward_vector + discount * np.dot(P, v))]
            vector_index += 1
    err = np.linalg.norm(v_old - v)
    print(err, n)

print(v)
print(pi)
# pi is the epsilon optimal policy
