from BayesNet import BayesNet
from BNReasoner import BNReasoner
import matplotlib.pyplot as plt
import networkx as nx
import itertools

#test_bn = BayesNet()
#test_bn.load_from_bifxml('testing\dog_problem.BIFXML')
#test_bn.load_from_bifxml('testing\covid.bifxml')

#reasoner = BNReasoner('testing\lecture_example.BIFXML')
reasoner = BNReasoner('testing\covid.BIFXML')
#reasoner.bn.draw_structure()
variables = reasoner.bn.get_all_variables()
#print(reasoner.bn.get_cpt('covid'))
#print(variables)
q = ['covid']
e = {'vaccinated': False}
# for var in variables:
#     print(var)
#     print(reasoner.marginal_dist([var]))
#     print("===============")
print(reasoner.marginal_dist(q,e))
e = {'vaccinated': True}
print(reasoner.marginal_dist(q,e))