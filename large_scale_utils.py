import numpy as np


cities = [i for i in range(1,21)]

# for many operations, dictionaries are one of the fastest data structures in python, and also the most readable
lams = {}
rhos = {}
demands = {}
m = {}
r = {}
c = {}

for city in cities:
    rhos[city] = np.random.uniform(0,1)

for city_i in cities:
    for city_j in cities:
        lams[city_i,city_j] = 2*rhos[city_i]*(1-rhos[city_j])
        m[city_i, city_j] = np.random.uniform(100, 1500)
        r[city_i, city_j] = rhos[city_i]*m[city_i, city_j]
        c[city_i, city_j] = 1.2*m[city_i, city_j]


# does the demand need to be generated each time epoch?
def generate_demands(lams, cities):
    demands = {}
    for city_i in cities:
        for city_j in cities:
            demands[city_i, city_j] = [np.random.poisson(lams[city_i, city_j])]
    return demands

print(m)
print(generate_demands(lams, cities))
T = 21