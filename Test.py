from BayesNet import BayesNet
from BNReasoner import BNReasoner
import matplotlib.pyplot as plt
import networkx as nx
import itertools

test_bn = BayesNet()
test_bn.load_from_bifxml('testing\dog_problem.BIFXML')
#test_bn.load_from_bifxml('testing\lecture_example.BIFXML')
#test_bn.draw_structure()

#reasoner = BNReasoner('testing\dog_problem.BIFXML')
reasoner = BNReasoner('testing\lecture_example.BIFXML')
#reasoner = BNReasoner('testing\lecture_example2.BIFXML')
#reasoner.bn.draw_structure()
#print([e for e in reasoner.bn.get_interaction_graph().edges])
#subax1 = plt.subplot(121)
#nx.draw(reasoner.bn.get_interaction_graph(), with_labels=True, font_weight='bold')
#plt.show()
# print("Min Degree order: ",reasoner.minDegreeOrder())
# print("Min Fill order: ", reasoner.minFillOrder())
#print(reasoner.d_sep(['bowel-problem', ['family-out'], ['light-on']]))
#print(reasoner.d_sep(['bowel-problem', ['dog-out'], ['hear-bark']]))
#reasoner.d_sep(['light-on', ['bowel-problem', 'dog-out'], ['family-out']])
# all_cpts = test_bn.get_all_cpts()
# for cpt in all_cpts:
#     print(cpt)
#     print(all_cpts[cpt])
#     print("-------------")
#q = ['light-on']
#e = {'dog-out': True, 'family-out': True}
q = ['Wet Grass?']
e = {'Winter?': True, 'Rain?': False}
reasoner.marginal_dist(q, e)
# bn = reasoner.network_pruning(q, e)
# all_cpts = bn.get_all_cpts()
# 
# for node in all_cpts:
#     print(bn.get_cpt(node))
# bn.draw_structure()