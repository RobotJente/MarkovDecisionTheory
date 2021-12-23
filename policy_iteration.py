#policy iteration
#POLICY_ITERATION
import copy

nmax=1000
#initialization
pi_0 = [actions[0] for x in range(16)]
pi=[None for x in range(16)]
pi=pi_0.copy()
n=0
converged = True

#iteration
while ( converged and (n<nmax )):
    pi_new=[None for x in range(16)]
    reward=np.zeros([16,1])    # immidiate reward vector for policy pi
    
    #for each state we compute the reward if the action  given by the policy in that state is taken
    vector_index=0
    for current_trailer_loc in nodes:
        for current_worker_loc in nodes:
            reward[vector_index]=calc_immediate_expected_reward(current_trailer_loc, current_worker_loc, pi[vector_index], transitions, action_results, nodes)
            vector_index+=1
            
    #once we have the reward vector we compute the trasition probability matrix for the policy
    matrix_row_index=0
    P=np.zeros([16,16])
    for current_trailer_loc in nodes:
        for current_worker_loc in nodes:
            matrix_column_index=0
            for future_trailer_loc in nodes:
                for future_worker_loc in nodes:
                    #if we move he trailer in a state the probability of going in a state in wich the trailer is in another site is 0
                    P[matrix_row_index][matrix_column_index]=(action_results[pi[matrix_row_index]]==future_trailer_loc)*transitions[nodes.index(current_worker_loc)][nodes.index(future_worker_loc)]
                    matrix_column_index+=1
            matrix_row_index+=1
    
    
    # Policy evaluation:
    #we compute the value of the policy
    v = np.dot(np.linalg.inv(np.eye(16) - 0.95*P), reward)
    
    
    # Policy improvement:
    #we have to find the best action for each state componentwise
    vector_index=0
    for current_trailer_loc in nodes:
        for current_worker_loc in nodes:
            #for each state we have 4 possible action, and so 4 possible reward
            reward_vector=np.zeros([4,1])
            P_componentwise=np.zeros([4,16])
            for idx,a in enumerate(actions):
                reward_vector[idx]=calc_immediate_expected_reward(current_trailer_loc, current_worker_loc, a, transitions, action_results, nodes)
                matrix_column_index=0
                for future_trailer_loc in nodes:
                    for future_worker_loc in nodes:
                        P_componentwise[idx][matrix_column_index]=(action_results[a]==future_trailer_loc)*transitions[nodes.index(current_worker_loc)][nodes.index(future_worker_loc)]
                        matrix_column_index+=1

                
            v2=reward_vector+discount*np.dot(P_componentwise,v)
            #if the best choice can be the same as before we choose that
            if(pi[vector_index] in set(actions[np.argmin(v2)])):
                pi_new[vector_index] = pi[vector_index]
            else: 
                pi_new[vector_index] = actions[np.argmin(v2)]
            vector_index+=1
            
    n=n+1
    if(pi==pi_new):
        converged=False
    else:
        pi=pi_new.copy()
#end