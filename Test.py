
from BNReasoner import BNReasoner

#reasoner = BNReasoner('testing\lecture_example.BIFXML')
reasoner = BNReasoner('testing\covid.BIFXML')
print(reasoner.bn.get_all_variables())
q = ['death']
e = {'vaccinated': False, 'covid': True, 'weak-immune': True, 'smoker':True}
print(reasoner.map(q,e))