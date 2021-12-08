from utils import *


# Parameters
discount = 0.95


# Made this string to avoid possible errors with integer multiplication (these should function as tokens, not integers)
states = ("Site 1", "Site 2", "Site 3", "Site 4")
actions = ("Move to site 1", "Move to site 2", "Move to site 3", "Move to site 4")

# encode a way to find the result of an action (in this case, the actions bring the trailer from site i to site j
action_results = {}
for i in range(len(actions)):
    action_results[actions[i]] = states[i]

