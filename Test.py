from BayesNet import BayesNet
from BNReasoner import BNReasoner
import matplotlib.pyplot as plt
import networkx as nx
import itertools

test_bn = BayesNet()
test_bn.load_from_bifxml('testing\dog_problem.BIFXML')
reasoner = BNReasoner('testing\lecture_example.BIFXML')

q = ['Wet Grass?']
e = {'Winter?': True, 'Rain?': False}
print(reasoner.marginal_dist(q, e))
