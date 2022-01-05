import numpy as np
from utils import calc_immediate_expected_reward
import copy

# value iteration algorithm for toy problem

# Find the probability of moving to a certain state "future_state", given that we are in state "current_state" and
# we have action "a" and transition matrix "transitions"
def prob_mat(current_state, future_state, a, transitions, action_results):
    current_worker_index = current_state[0]-1
    future_worker_index = future_state[0]-1

    if action_results[a] == future_state[1]:
        return transitions[current_worker_index][future_worker_index]
    else:
        return 0

# Makes the transition matrix for the states S x S, i.e. the tuple pair of (worker_location, trailer_location), based on a policy
def make_p(transitions, policy, actions, action_results, nodes):
    num_states = len(nodes)
    p = np.zeros([num_states**2, num_states**2])

    # double loop to explore each permutation of workers and trailers
    for worker_index in range(num_states):
        for trailer_index in range(num_states):
            for future_worker in range(num_states):
                for future_trailer in range(num_states):
                    p[worker_index + future_worker][trailer_index + future_trailer] = prob_mat((worker_index, trailer_index), (future_worker, future_trailer), actions[worker_index][trailer_index], transitions, action_results)
    return p


def perform_val_it(transitions, actions, nodes, discount, epsilon, action_results):
    # initialize v for each state
    err = 1
    dim_states = transitions.shape[0]
    pi = {}  # [None for i in range(len(v))]  # here we will save the policies
    v = np.ones([dim_states ** 2, 1])
    helper_v = {}
    n = 0



    # initialize helper dictionary
    for init_state_worker in nodes:
        for init_state_trailer in nodes:
            helper_v[(init_state_worker, init_state_trailer)] = 1

    # value iteration algorithm
    print(helper_v, "INIT")
    while (err > epsilon * (1 - discount) / (2 * discount)):
        n += 1

        # track old value to determine convergence. use deepcopy since python normally tracks the reference, not the value
        v_old = copy.deepcopy(v)
        helper_v_old = copy.deepcopy(helper_v)

        # state_index corresponds to the current state being evaluated, and runs from 0 to len(nodes)**2
        state_index = 0
        for current_worker_loc in nodes:
            for current_trailer_loc in nodes:
                reward_vector = []

                for action in actions:
                    r = calc_immediate_expected_reward(current_trailer_loc, current_worker_loc, action, transitions,
                                                       action_results, nodes)
                    for worker_loc in nodes:
                        for trailer_loc in nodes:
                            r += discount * prob_mat((current_worker_loc, current_trailer_loc),
                                                     (worker_loc, trailer_loc), action, transitions, action_results) * \
                                 helper_v_old[(worker_loc, trailer_loc)]
                    reward_vector.append(r)

                # For each state, choose v to be the result of the action with the highest reward
                v[state_index] = np.min(reward_vector)
                # store the copies of v in a dictionary so it's easier to access specific states
                helper_v[(current_worker_loc, current_trailer_loc)] = copy.deepcopy(v[state_index])

                print("Helper v", helper_v)
                print(v)
                # we have now traversed a state
                state_index += 1
        # track the convergence
        err = np.linalg.norm(v_old - v)
        print(err, n)

    # after convergence, traverse each state, and for each state,
    # find the best action, using the last value of v from the iterations

    for current_trailer_loc in nodes:
        for current_worker_loc in nodes:
            reward_vector = []
            for action in actions:
                r = calc_immediate_expected_reward(current_trailer_loc, current_worker_loc, action, transitions,
                                                   action_results, nodes)
                for worker_loc in nodes:
                    for trailer_loc in nodes:
                        r += discount * prob_mat((current_worker_loc, current_trailer_loc), (worker_loc, trailer_loc),
                                                 action, transitions, action_results) * helper_v[
                                 (worker_loc, trailer_loc)]
                reward_vector.append(r)

            # For each state, choose v to be the result of taking the action in that state with the highest reward
            pi[(current_worker_loc, current_trailer_loc)] = actions[np.argmin(reward_vector)]

            # we have now traversed a state
    print("optimal policy is given by \n", pi)



