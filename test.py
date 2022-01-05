#value iteration algorithm

#initialization
epsilon=1e-5
discount = 0.95

#initialize v for each state
v = {}
v[nodes[0]] = (1, 1, 1, 1)
v[nodes[1]] = (1, 1, 1, 1)
v[nodes[2]] = (1, 1, 1, 1)
v[nodes[3]] = (1, 1, 1, 1)

err=1

n=0
while(err>epsilon*(1-discount)/(2*discount)):
    n=n+1
    v_old=v
    for trailerstate, workerstate in nodes:
        R={}

        for a in actions:
            if a==trailestates:
                R[a]=calc_total_expected_reward(workerstate, trailerstate, action, transitions, action_results, nodes)
            else:
                R[a]=calc_total_expected_reward(workerstate, trailerstate, action, transitions, action_results, nodes)
                for i in range(len(nodes)):
                    R[a]+=discount*transitions[current_worker_location][i]*v[trailerstate][state.index(workstate)]
        v[trailerstate,[states.index(workstate)]]=max(R)
    err=norm(v_old-v)






