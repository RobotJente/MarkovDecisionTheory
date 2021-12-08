
# return the cost for moving the trailer between locations
def calc_cost_move_trailer(current_loc, future_loc):
    if current_loc is not future_loc:
        return 300
    else:
        return 0

# return the cost for using the trailer if the workers are at location worker_loc, and the trailer is at
# location trailer_loc
def calc_cost_use_trailer(worker_loc, trailer_loc, states):
    # If workers are at home location, no work done, so cost 0 regardless of trailer location
    if worker_loc == states[0]:
        return 0
    # If trailer at home and work force is not, then it costs 200
    if trailer_loc is not states[0]:
        if worker_loc == trailer_loc:
            return 50
        else:
            return 100
    else:
        return 200

# given an action, find the total expected reward of this action
def calc_total_expected_reward(current_state, action, transitions, action_results):
    # find direct costs of the action (moving the trailer)
    cost = 0
    future_trailer_loc = action_results[action]
    cost += calc_cost_move_trailer(current_loc=current_state, future_loc= future_trailer_loc)

    # find expected rewards of the action by using the transition probabilities
    for i in enumerate(transitions):
        # sum of expected rewards: prob of moving to state i * cost of trailer at state i
        cost += calc_cost_use_trailer(worker_loc=transitions[i], trailer_loc=future_trailer_loc)*transitions[current_state][0]

# for example, if we are in state 4, we cannot go to state 2 and 3, we can only go to state 2 or stay in state 4
def build_transition_matrix(states):
    transitions = {}
    transitions[states[0]] = (0.1, 0.3, 0.3, 0.3)
    transitions[states[1]] = (0, 0.5, 0.5, 0)
    transitions[states[2]] = (0, 0, 0.8, 0.2)
    transitions[states[3]] = (0.4, 0, 0, 0.6)
    return transitions
