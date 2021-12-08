#value iteration algorithm

#initialization
epsilon=1e-5
discount = 0.95

#initialize v for each state
v = {}
v[states[0]] = (1, 1, 1, 1)
v[states[1]] = (1, 1, 1, 1)
v[states[2]] = (1, 1, 1, 1)
v[states[3]] = (1, 1, 1, 1)

for trailerstate, workerstate in states:
    R={}
    for a in actions:
        if a==trailestates:
            R[a]=calc_total_expected_reward(workerstate, trailerstate, action, transitions, action_results, states)
        else:
            R[a]=calc_total_expected_reward(workerstate, trailerstate, action, transitions, action_results, states)
            for i in len(states):
                R[a]+=discount*transitions[current_worker_location][i]*v[trailerstate][workstate]
    v[trailerstate,workstate]=max(R)


