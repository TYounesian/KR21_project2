from typing import Union
from BayesNet import BayesNet
import pdb

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go

    def dSep(self, x, y, z):
        """
        Checks to see if x and y are independent with respect to z
        :param variables: variables to check independence with the given order

        """ 

        net_variables = self.bn.get_all_variables()
        w = [var for var in net_variables if var not in [x,y,z]]
        
        leaf_nodes = []

        for var in range (len(w)):
            vars_children = self.bn.get_children(w[var])
            if vars_children == []:
                leaf_nodes.append(w[var])

        pruned_net = self.bn
        for i in range(len(leaf_nodes)):
            pruned_net.del_var(leaf_nodes[i])    
        

        z_children = self.bn.get_children(z)
        for j in range(len(z_children)):
            pruned_net.del_edge(z, z_children[j])





test_net = BNReasoner('testing/dog_problem.BIFXML')
test_net.bn.draw_structure()
test_net.dSep('bowel-problem', 'family-out', 'dog-out')
# pdb.set_trace()