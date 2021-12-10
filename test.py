from typing import Union
from BayesNet import BayesNet
import pdb
import copy

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

        pruned_net = copy.deepcopy(self.bn)
        
        for i in range(len(leaf_nodes)):
            pruned_net.del_var(leaf_nodes[i])  # node pruning  
        
        z_children = pruned_net.get_children(z)
        for j in range(len(z_children)):
            pruned_net.del_edge([z, z_children[j]]) # edge pruning
        pruned_net.draw_structure()
        # Check if the network is separated 
        
        vars_children = []
        pruned_net_vars = pruned_net.get_all_variables()
        for var in range(len(pruned_net_vars)):
            vars_children.append(pruned_net.get_children(pruned_net_vars[var]))

            if pruned_net.get_children(pruned_net_vars[var]) == []: # nodes without childeren
                #pdb.set_trace()
                for i in range(len(pruned_net_vars)): # check if that node appears in any other node's children list
                    if (pruned_net_vars[var] in set(pruned_net.get_children(pruned_net_vars[i]))): # it's some node's child
                        vars_children = []
                        break # go to the next node
                if not(pruned_net_vars[var] in set(pruned_net.get_children(pruned_net_vars[i]))): #it's no node's child
                    print ("Network separated") 
                    break




test_net = BNReasoner('testing/dog_problem.BIFXML')
test_net.bn.draw_structure()
test_net.dSep('bowel-problem', 'dog-out', 'family-out')
# pdb.set_trace()