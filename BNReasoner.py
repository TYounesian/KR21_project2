from typing import Union
from BayesNet import BayesNet
import sys
from copy import deepcopy
import itertools
import matplotlib.pyplot as plt
import networkx as nx


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


    def d_sep(self, variables):
        """
        D-separation implies variables X and Y are independent given Z
        Two rules are followed:
        - Delete every leaf node W that is not part of X U Y U Z
        - Delete all edges outgoing from nodes in Z
        :arg variables: list of variables in the format (X, Y, Z)
        :returns: True if X and Y are d-separated by Z, and False otherwise
        """
        if len(variables) != 3:
            print("Please input three sets of variables.")
            sys.exit()

        bn = deepcopy(self.bn)
        variables = [x if isinstance(x, list) else [x] for x in variables]  # Making sure the sets are lists
        x, z, y = variables  # Assign the variables to be examined

        # Delete leaf nodes
        leaf_nodes = self.get_leaf_nodes(bn)
        for var in leaf_nodes:
            if var not in x + y + z:
                bn.del_var(var)
        # Delete edges from z
        for var in z:
            for children in bn.get_children(var):
                bn.del_edge([var, children])
        while True:  # Create an iterative loop
            leaf_nodes = self.get_leaf_nodes(bn)
            for var in leaf_nodes:
                if var not in x + y + z:
                    bn.del_var(var)
            if not set(self.get_leaf_nodes(bn)) - set(x + y + z):  # Break when only leaf nodes are in x + y + z
                break
        # Path searching:
        return self.bfs(bn, x, y)


    def get_leaf_nodes(self, net):
        """
        Checks for the variables that do not have any children
        :arg: Bayesian network
        :returns: a list of variable names that are leaf nodes
        """
        return [var for var in net.get_all_variables() if not net.get_children(var)]

    def delete_edges_from(self, variables, net):
        """
        Deletes the edges starting out from the variable
        :arg variables: The variable that needs to have its edges removed
        :arg net: Bayesian Network
        """
        for var in variables:
            for children in net.get_children(var):
                net.del_edge([var, children])
        return net

    def bfs(self, net, starting_nodes, ending_nodes):
        """
        Breadth-first search to find a path from starting_node to ending_node
        :arg net: Bayesion Network
        :param starting_nodes: The node we want to start the search from the (list)
        :param ending_nodes: A set of nodes we want to find the path to
        :return False if the path is found, True otherwise
        """
        visited = []
        queue = starting_nodes
        while queue:
            node = queue.pop(0)
            if node in ending_nodes:
                return False
            if node not in visited:
                visited.append(node)
                neighbors = itertools.chain(net.structure.neighbors(node), net.structure.predecessors(node))
                for neighbor in neighbors:
                    queue.append(neighbor)
        return True

    def minDegreeOrder(self, bn=None):
        """
        Always chooses the smallest degree (smallest width - smallest number of neighbors)
        :returns an ordering pi of variables of the BN
        """
        if not bn:
            bn = self.bn
        g = bn.get_interaction_graph()
        x = bn.get_all_variables()
        pi = []
        # for node in x:
        #     print(node, g.adj[node].items())
        print([e for e in g.edges])
        for i in range(len(x)):
            n_neighbours = [len(g.adj[node].items()) for node in x]
            pi.append(x[n_neighbours.index(min(n_neighbours))])
            # Add edge between every pair of non-adjacent neighbors of pi(i) in G
            neighbors = [s for s, _ in g.adj[pi[i]].items()] + [pi[i]]
            for subset in itertools.combinations(neighbors, 2):
                if not g.has_edge(*subset):
                    g.add_edge(*subset)
            # delete variable pi(i) from G and from X
            g.remove_node(pi[i])
            x.remove(pi[i])
        return pi

    def minFillOrder(self, bn=None):
        """
        Always chooses the node whose elimination adds the smallest number of edges
        :returns an ordering pi of variables of the BN
        """
        if not bn:
            bn = self.bn
        g = bn.get_interaction_graph()
        x = bn.get_all_variables()
        pi = []
        for i in range(len(x)):
            n_edges = []
            for node in x:
                n = [s for s, _ in g.adj[node].items()]
                if len(n) == 1:
                    n_edges.append(0)
                else:
                    n_edges.append(sum([0 if g.has_edge(*subset) else 1 for subset in itertools.combinations(n, 2)]))
            pi.append(x[n_edges.index(min(n_edges))])
            # Add edge between every pair of non-adjacent neighbors of pi(i) in G
            neighbors = [s for s, _ in g.adj[pi[i]].items()] + [pi[i]]
            for subset in itertools.combinations(neighbors, 2):
                if not g.has_edge(*subset):
                    g.add_edge(*subset)
            # delete variable pi(i) from G and from X
            g.remove_node(pi[i])
            x.remove(pi[i])
        return pi

    def order_width(self, bn, pi):
        """
        :param bn: Bayesian Network
        :param pi: ordering of the variables in network bn
        :return: The width of elimination order pi
        """
        w = 0
        g = bn.get_interaction_graph()
        for i in range(len(pi)):
            w = max(w, len(g.adj[pi[i]].items()))
            # Add edge between every pair of non-adjacent neighbors of pi(i) in G
            for subset in itertools.combinations(g.adj[pi[i]].items(), 2):
                if not g.has_edge(*subset):
                    g.add_edge(*subset)
            # delete variable pi(i) from G
            g.remove_node(pi[i])
        return w

    def network_pruning(self, q, e):
        """
        :param q: list of values the query addresses
        :param e: evidence (dict of variables and their values)
        :returns: a pruned bayesian network
        """
        bn = deepcopy(self.bn)
        # Remove edges
        for var in e.keys():
            for children in bn.get_children(var):
                bn.del_edge([var, children])  # Remove edges
                # Update CPT
                cpt = bn.get_cpt(children)
                bn.update_cpt(children, cpt[cpt[var] == e[var]].drop(columns=var))
        # Remove leaf nodes once again
        while True:  # Create an iterative loop
            leaf_nodes = self.get_leaf_nodes(bn)
            for var in leaf_nodes:
                if var not in q + list(e.keys()):
                    bn.del_var(var)
            if not set(self.get_leaf_nodes(bn)) - set(q + list(e.keys())):  # Break when only leaf nodes are in x + y + z
                break
        return bn

    def marginal_dist(self, q, e={}):
        """
        Calculates the marginal probability of the set of variables q, given possible evidence e
        :returns: CPT for the set of variable q
        """
        bn = self.network_pruning(q, e)
        s = bn.get_all_cpts()
        pi = self.minFillOrder(bn)
        #nx.draw(bn.get_interaction_graph(), with_labels=True, font_weight='bold')
        #plt.show()
        #bn.draw_structure()
        for node in pi:
            mentions = [v for v in s.values() if node in v.columns and not v.equals(s[node])]
            if len(mentions):
                for i in range(len(mentions)):
                    mentions[i]
                false = s[node].loc[s[node][node] == False]['p']
                true = s[node].loc[s[node][node] == True]['p']



            # if len(toMultiply) != 1:
            #     false = s[node].loc[s[node][node] == False]['p']
            #     true = s[node].loc[s[node][node] == True]['p']
            #
            #     print()
            #     print("-------------")
        return "Over"
                #print(v.columns)
           # print(node)
            # Then the summation


    def map(self):
        pass

    def mpe(self):
        pass
