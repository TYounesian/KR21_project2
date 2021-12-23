from BNReasoner import BNReasoner
import pandas as pd
import random
import time
from BayesNet import BayesNet
import time

#reasoner = BNReasoner('testing\lecture_example.BIFXML')
reasoner = BNReasoner('testing\covid.BIFXML')
q = ['covid']
e = {'death': True, 'tests-match': False}
# print(reasoner.bn.get_all_variables())
# q = ["Wet Grass?"]
# e = {"Winter?": True, "Rain?": False}
pr = reasoner.marginal_dist(q,e)
# for cpt in pr:
#     print(pr[cpt])
#print(reasoner.mpe(e))
#print(reasoner.map(q, e, ordering="minFill"))

#Generate network
network_size = 100
leaf_nodes = reasoner.get_leaf_nodes(reasoner.bn)
print("Generating network")
for i in range(network_size):
    new_node = "node_" + str(i)
    evidence_vars = random.choices(reasoner.bn.get_all_variables(), k=random.choice([1, 2, 3]))

    leaf_node = reasoner.get_leaf_nodes(reasoner.bn)[0]
    cpt = pd.DataFrame({leaf_node : [True, False, True, False],
                        new_node : [True, True, False, False],
                        'p' : [0.1, 0.9, 0.2, 0.8]})
    reasoner.bn.add_var(new_node, cpt=cpt)
    reasoner.bn.add_edge((leaf_node, new_node))
all_wars = reasoner.bn.get_all_variables()
print("Staring measurements")
n_vars = 1
minFill = {}
for i in range(1, 100):
    evidence_vars = random.choices(all_wars, k=n_vars)
    evidence = {}
    for v in evidence_vars:
        evidence[v] = random.choice([False, True])
    query_var = random.choices(list(set(all_wars) - set(evidence_vars)), k=n_vars)
    start = time.time()
    reasoner.map(query_var, evidence)
    end = time.time()
    timing = end - start
    minFill[n_vars] = timing
    n_vars = i * 10
    print(minFill)

#print(reasoner.map(q,e))
#print(reasoner.bn)
# reasoner = BNReasoner('testing\lecture_example2.BIFXML')
# Classrooom example for MPE
# e = {'J' :True, 'O':False}
# print(reasoner.mpe(e))
# Classrooom example for MAP
# q = ['I', 'J']
# e = {'O' :True}
# print(reasoner.map(q, e))