
# return the cost for moving the trailer between locations (d(k,a)=calc_cost_move_trailer(current_loc, future_loc))
def calc_cost_move_trailer(current_trailer_loc, future_trailer_loc):
    if current_trailer_loc is not future_trailer_loc:
        return 300
    else:
        return 0

# return the cost for using the trailer if the workers are at location worker_loc, and the trailer is at
# location trailer_loc
def calc_cost_use_trailer(worker_loc, trailer_loc, nodes):
    # If workers are at home location, no work done, so cost 0 regardless of trailer location
    if worker_loc == nodes[0]:
        return 0
    # If trailer at home and work force is not, then it costs 200
    if trailer_loc is not nodes[0]:
        if worker_loc == trailer_loc:
            return 50
        else:
            return 100
    else:
        return 200

# for each state (current_trailer_location, current_worker_location) and for each action compute the expercted immediate reward
def calc_immediate_expected_reward(current_trailer_loc, current_worker_loc, action, workers_transition_probability, action_results, nodes):
    # find direct costs of the action (moving the trailer)
    cost = 0
    future_trailer_loc = action_results[action]
    cost += calc_cost_move_trailer(current_trailer_loc=current_trailer_loc, future_trailer_loc= future_trailer_loc)

    # find expected rewards of the action by using the transition probabilities
    for i in range(len(nodes)):
        # sum of expected rewards: prob of moving to state i * cost of trailer at state i
        cost += calc_cost_use_trailer(worker_loc=nodes[i], trailer_loc=future_trailer_loc, nodes = nodes)*workers_transition_probability[nodes.index(current_worker_loc)][i]
    return cost

#this is useless now
# # for example, if we are in state 4, we cannot go to state 2 and 3, we can only go to state 2 or stay in state 4
# def build_transition_matrix(nodes):
#     transitions = {}
#     transitions[nodes[0]] = (0.1, 0.3, 0.3, 0.3)
#     transitions[nodes[1]] = (0, 0.5, 0.5, 0)
#     transitions[nodes[2]] = (0, 0, 0.8, 0.2)
#     transitions[nodes[3]] = (0.4, 0, 0, 0.6)
#     return transitions
