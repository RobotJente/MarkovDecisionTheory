import numpy as np
from scipy import linalg
import numpy as np
from random import seed
from random import random
import matplotlib.pyplot as plt

transitions = np.zeros([4,4])
transitions[0] = [0.1, 0.3, 0.3, 0.3]
transitions[1] = [0, 0.5, 0.5, 0]
transitions[2] = [0, 0, 0.8, 0.2]
transitions[3] = [0.4, 0, 0, 0.6]
lefts = linalg.eig(transitions, left = True, right = False)

for elem in lefts:
    a = elem@transitions
    print(np.linalg.norm(a-elem))


A=np.append(np.transpose(transitions)-np.eye(4),[[1,1,1,1]],axis=0)
b=np.transpose(np.array([0,0,0,1,1]))
x = np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(b))
print(x)