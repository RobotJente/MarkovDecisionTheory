from utils import *
import numpy as np


#MODEL
nodes = ("Site 1", "Site 2", "Site 3", "Site 4")
actions = ("Move to site 1", "Move to site 2", "Move to site 3", "Move to site 4")  # Made this string to avoid possible errors with integer multiplication (these should function as tokens, not integers)

#probability_worker-transition_matrix
transitions = np.zeros(shape=(4,4))
transitions[0] = [0.1, 0.3, 0.3, 0.3]
transitions[1] = [0, 0.5, 0.5, 0]
transitions[2] = [0, 0, 0.8, 0.2]
transitions[3] = [0.4, 0, 0, 0.6]
# Parameters
discount = 0.95

#the functions that calculates rewards and transition probability matrix are in utils


#value iteration algorithm

#initialization
epsilon=1e-5

#initialize v for each state
v = np.ones(16,1)
err=1
pi=[]


n=0
while(err>epsilon*(1-discount)/(2*discount)):
    n=n+1
    v_old=v
    vector_index=0
    for current_trailer_loc in nodes:
        for current_worker_loc in nodes:
            reward_vector=np.zeros(4,1)
            #probability matrix for that action
            P=np.zeros(4,16)
            for idx,a in eumerate(action):
                reward_vector[idx]=calc_immediate_expected_reward(current_trailer_loc, current_worker_loc, a, transitions, action_results, nodes)
                matrix_column_index=0
                for future_trailer_loc in nodes:
                    for future_worker_loc in nodes:
                        P[idx][matrix_column_index]=(action_results[a]==future_trailer_location)*transitions[nodes.index(current_worker_loc)][nodes.index(future_worker_loc)]
                        matrix_column_index+=1
            v[vector_index]=np.max(reward_vector+discount*np.dot(P,v))
            pi[vector_index]=action[np.argmax(reward_vector+discount*np.dot(P,v))]
            vector_index+=1
    err=np.norm(v_old-v)


print(v)
print(pi)
#pi is the epsilon optimal policy




#POLICY_ITERATION

#we have to initialize a vector of 16 entries that represents our vale (total expected reward)
v=np.ones(16,1)

# encode a way to find the result of an action (in this case, the actions bring the trailer from site i to site j
action_results = {}
for i in range(len(actions)):
    action_results[actions[i]] = nodes[i]

#initialize somethi g
pi_0 = [action[0] for x in range(16)]


# evaluate the policy pi by calculating the state value function V
converged = False
pi=pi_0
while not converged:
    pi_old=pi
    reward=np.zeros(16,1)    # reward vec comprises of only the short term rewards
    #for each state i have to compute the reward if I take the action theta i given by the policy in that state
    vector_index=0
    for current_trailer_loc in nodes:
            for current_worker_loc in nodes:
                reward[vector_index]=calc_immediate_expected_reward(current_trailer_loc, current_worker_loc, pi[index_vector], transitions, action_results, nodes)
                vector_index+=1
    #ones i have my reward vector i have to compute the trasition probability matrix
    matrix_row_index=0
    P=np.zeros(16,16)
    #for cycle foe each current state
    for current_trailer_loc in nodes:
            for current_worker_loc in nodes:
                matrix_column_index=0
                #for cycle for each future state
                for future_trailer_loc in nodes:
                    for future_worker_loc in nodes:
                        P[matrix_row_index][matrix_column_index]=(action_results[pi[matrix_row_index]]==future_trailer_location)*transitions[nodes.index(current_worker_loc)][nodes.index(future_worker_loc)]
                        matrix_column_index+=1
                matrix_row_index+=1
    #I built our transition matrix
    # Policy evaluation:
    v = np.dot(np.linalg.inv(np.eye(16) - 0.95*P), reward)


    #this is the value of our policy
    # Policy improvement:
    #i haev to find the best action for each state componentwise
    vector_index=0
    for current_trailer_loc in nodes:
            for current_worker_loc in nodes:
                #for each state I have 4 possible action, and so 4 possible reward
                reward_vector=np.zeros(4,1)
                P_componentwise=np.zerso(4,16)
                for idx,a in eumerate(action):
                    reward_vector[idx]=calc_immediate_expected_reward(current_trailer_loc, current_worker_loc, a, transitions, action_results, nodes)
                    matrix_column_index=0
                    for future_trailer_loc in nodes:
                        for future_worker_loc in nodes:
                            P_componentwise[idx][matrix_column_index]=(action_results[a]==future_trailer_location)*transitions[nodes.index(current_worker_loc)][nodes.index(future_trailer_loc)]
                            matrix_column_index+=1

                
                v2=reward_vector+discount*np.dot(P_componentwise,v)
                pi[vector_index]=action[np.argmax(v2)]
                vector_index+=1
    if (pi==pi_old):
        converged = True

#end
    
