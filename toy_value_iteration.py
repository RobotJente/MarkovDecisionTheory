import numpy as np
from utils import calc_immediate_expected_reward
from main import action_results
import copy

# Find the probability of moving to a certain state "future_state", given that we are in state "current_state" and
# we have action "a" and transition matrix "transitions"
def prob_mat(current_state, future_state, a, transitions, action_results):
    current_worker_index = current_state[0]-1
    future_worker_index = future_state[0]-1

    if action_results[a] == future_state[1]:
        return transitions[current_worker_index][future_worker_index]
    else:
        return 0

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
# value iteration algorithm for toy problem

# initialization
epsilon = 1e-9
discount = 0.95


transitions = np.zeros(shape=(2,2))
transitions[0] = [0.99, 0.01]
transitions[1] = [0.99, 0.01]

dim_states = transitions.shape[0]
# initialize v for each state
v = np.ones([dim_states**2, 1])
err = 1
pi = [None for i in range(len(v))]  # here we will save the policies
nodes = [1, 2]
actions = ("Move to site 1", "Move to site 2")
n = 0
# encode a way to find the result of an action (in this case, the actions bring the trailer from site i to site j
action_results = {}
for i in range(len(actions)):
    action_results[actions[i]] = i+1


while (err > epsilon * (1 - discount) / (2 * discount)):
    n += 1

    # track old value to determine convergence. use deepcopy since python normally tracks the reference, not the value
    v_old = copy.deepcopy(v)

    # state_index corresponds to the current state being evaluated. For example, state_index=5 corresponds to the state (3,2) in this problem)
    state_index = 0
    for current_trailer_loc in nodes:
        for current_worker_loc in nodes:
            reward_vector = []

            for action in actions:
                r = calc_immediate_expected_reward(current_trailer_loc, current_worker_loc, action, transitions, action_results, nodes)
                for worker_loc in nodes:
                    for trailer_loc in nodes:
                        r += discount*prob_mat((current_worker_loc, current_trailer_loc), (worker_loc, trailer_loc), action, transitions, action_results)*v_old[state_index]
                reward_vector.append(r)

            # For each state, choose v to be the result of the action with the highest reward
            v[state_index] = np.max(reward_vector)

            # we have now traversed a state
            state_index += 1

    err = np.linalg.norm(v_old - v)
    print(err, n)

state_index = 0
for current_trailer_loc in nodes:
    for current_worker_loc in nodes:
        reward_vector = []
        for action in actions:
            r = calc_immediate_expected_reward(current_trailer_loc, current_worker_loc, action, transitions,
                                               action_results, nodes)
            for worker_loc in nodes:
                for trailer_loc in nodes:
                    print(v[state_index])
                    r += discount * prob_mat((current_worker_loc, current_trailer_loc), (worker_loc, trailer_loc),
                                             action, transitions, action_results) * v[state_index]
            reward_vector.append(r)
        print(reward_vector, "reward vec")
        # For each state, choose v to be the result of the action with the highest reward
        pi[state_index] = actions[np.argmax(reward_vector)]

        # we have now traversed a state
        state_index += 1

print(pi)



print(v)
print(pi)
# pi is the epsilon optimal policy


# print(prob_mat([1,1], [2,2], actions[0], transitions, action_results)) # from (1,1) to (2,2) is not possible because the action moves the trailer to state (i, 1)
# print(prob_mat([1,1], [2,3], actions[1], transitions, action_results)) # from (1,1) to (2,2) is not possible because the action moves the trailer to state (i, 1)
# print(prob_mat([1,1], [2,2], actions[0], transitions, action_results)) # from (1,1) to (2,2) is not possible because the action moves the trailer to state (i, 1)
# print(prob_mat([1,1], [2,2], actions[1], transitions, action_results)) # from (1,1) to (2,2) is not possible because the action moves the trailer to state (i, 1)
