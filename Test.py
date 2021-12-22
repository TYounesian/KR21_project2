from BNReasoner import BNReasoner
import pandas as pd
import random
import time
from BayesNet import BayesNet

#reasoner = BNReasoner('testing\lecture_example.BIFXML')
#reasoner = BNReasoner('testing\car-starts.xml')
reasoner = BNReasoner('testing\covid.BIFXML')

q = ['covid']
e = {'death': True, 'tests-match': True}
#print(reasoner.map(q, e))
# print("Old implemenatation:")
# print(reasoner.mpe(e))
print("\nNew implemenatation:")
print(reasoner.mpe2(e, ordering="minFill"))
# Generate network
# network_size = 0
# leaf_nodes = reasoner.get_leaf_nodes(reasoner.bn)
# print("Generating network")
# for i in range(network_size):
#     node_name = "node_" + str(i)
#     leaf_node = reasoner.get_leaf_nodes(reasoner.bn)[0]
#     cpt = pd.DataFrame({leaf_node : [True, False, True, False],
#                         node_name : [True, True, False, False],
#                         'p' : [0.1, 0.9, 0.2, 0.8]})
#     reasoner.bn.add_var(node_name, cpt=cpt)
#     reasoner.bn.add_edge((leaf_node, node_name))
# all_wars = reasoner.bn.get_all_variables()
# print("Staring measurements")
# n_vars = 1
# minFill = {}
# for i in range(5):
#     evidence_vars = random.choices(all_wars, k=n_vars)
#     evidence = {}
#     for v in evidence_vars:
#         evidence[v] = random.choice([False, True])
#     print(evidence)
#     query_var = random.choices(list(set(all_wars) - set(evidence_vars)), k=n_vars)
#     print(query_var)
#     start = time.time()
#     reasoner.map(query_var, evidence)
#     end = time.time()
#     timing = end - start
#     minFill[n_vars] = timing
#     n_vars += i
#     print(minFill)

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