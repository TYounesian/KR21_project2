from typing import Union
from BayesNet import BayesNet
import sys
from copy import deepcopy
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


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
        self.options = [True, False]

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
            cpt = bn.get_cpt(var)
            bn.update_cpt(var, cpt[cpt[var] == e[var]])
            for children in bn.get_children(var):
                bn.del_edge([var, children])  # Remove edges
                # Update CPT
                cpt = bn.get_cpt(children)
                bn.update_cpt(children, cpt[cpt[var] == e[var]])#.drop(columns=var))
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
        print("\nCalculating marginal distribution of {} given evidence {}.\n".format(q, e))
        bn = self.network_pruning(q, e)
        s = bn.get_all_cpts()
        pi = self.minDegreeOrder(bn)
        for node in q:
            pi.remove(node)

        for i in range(len(pi)):
            mentions = {}
            for cp, cpt in s.items():
                if pi[i] in cpt.columns:
                    mentions[cp] = cpt
            # Multiply the cpts that mention the node
            f = self.eliminate_variable(mentions.values(), pi[i], e)
            # Replace the cpt-s with the new factor
            s['f'+str(i)] = f
            for k in mentions.keys():
                del s[k]
        # Normalise by evidence:
        if e:
            for cpt in s.values():
                for k, v in e.items():
                    e_cpt = bn.get_cpt(k)
                    e_cpt = e_cpt[e_cpt['p'] != 0]
                    toDivide = float(e_cpt[e_cpt[k] == v]['p'])
                    cpt['p'] /= toDivide

        return s

    def eliminate_variable(self, cpts, pi_i, evidence = {}):
        """
        :arg: cpts: The conditional probability tables (factors) to multiply
        :return a factor corresponding to the product
        """
        # Initialize a dataframe Z
        z_cols = [c for c in pd.concat(cpts, axis=0, ignore_index=True).columns if c != 'p']
        z = pd.DataFrame(list(itertools.product(*[self.options for i in range(len(z_cols))])), columns=z_cols)
        z['p'] = [1] * z.shape[0]
        z = self.exclude_evidence(z, evidence)
        # Find rows of z that are consistent with the cpt and multiply
        print(type(cpts))
        print(cpts)
        print("\n***\n")
        for cpt in cpts:
            for i, row_content in cpt.iterrows():
                for i_z, row_content_z in z.iterrows():
                    if all([row_content_z[c] == row_content[c] for c in cpt.columns if c != 'p']):
                        z.iloc[i_z, -1] *= row_content['p']
        # Summing out pi(i)
        z.drop(columns=[pi_i], inplace=True)
        cols = [col for col in z.columns if col != 'p']
        toAdd = set()
        for i, row in z.iterrows():
            toCheck = [row[col] for col in cols]
            for i2, row2 in z.iterrows():
                toCompare = [row2[col] for col in cols]
                if toCheck == toCompare and i2 != i and (i2, i) not in toAdd:
                    toAdd.add((i, i2))
        for i in range(len(toAdd)):
            z.iloc[list(toAdd)[i][0], -1] += z.iloc[list(toAdd)[i][1], -1]
        for i in range(len(toAdd)):
            z.drop(list(toAdd)[i][1], inplace=True)
        z.reset_index(inplace=True, drop=True)
        return z


    def map(self, vars, e={}, ordering="minFill"):
        """
        Maximises the value and instantiation of variables m given evidence e
        :return instantiation and value
        """
        # Get the elimination order
        if ordering == "minDegree":
            pi = self.minDegreeOrder()
        else:
            pi = self.minFillOrder()
        # Create a single table based on pi (and possible evidence e)
        z = self.create_table()
        z = self.exclude_evidence(z, e)
        # Multiply all the variables
        to_multiply = [self.bn.get_cpt(col) for col in pi]
        z = self.multiply_factors(to_multiply, z)
        # Sum out all the variables non-Map
        pi = [c for c in pi if c not in vars]
        z = self.summing_out(pi, z)
        print(z)
        # Max out the MAP variables one-by-one according to pi
        # Return the most likely instantiation and it's probability
        return z.iloc[z['p'].idxmax()]

    def mpe(self, e={}, ordering="minFill"):
        """
        Finds the Most Probable Explanation given possible evidence e
        :return instantiation and value
        """
        """
        Maximises the value and instantiation of variables m given evidence e
        :return instantiation and value
        """
        # Get the elimination order
        if ordering == "minDegree":
            pi = self.minDegreeOrder()
        else:
            pi = self.minFillOrder()
        # Create a single table based on pi (and possible evidence e)
        z = self.create_table()
        z = self.exclude_evidence(z, e)
        # Multiply all the variables
        to_multiply = [self.bn.get_cpt(col) for col in pi]
        z = self.multiply_factors(to_multiply, z)
        # Max out variables one-by-one according to pi
        max_prob = self.maxing_out(pi, z)
        # Return the most likely instantiation and it's probability
        return max_prob

    def multiply_factors(self, cpts, z):
        """
        Multiplies the corresponding rows of the cpts and z
        :return z: a cpt containing all the p-values of the cpts
        """
        for cpt in cpts:
            for i, row_content in cpt.iterrows():
                for i_z, row_content_z in z.iterrows():
                    if all([row_content_z[c] == row_content[c] for c in cpt.columns if c != 'p']):
                        z.iloc[i_z, -1] *= row_content['p']
        return z

    def summing_out(self, ordering, cpt):
        """
        :return A cpt with variables in ordering summed out
        """
        for var in ordering:
            cpt.drop(columns=[var], inplace=True)
            cols = [col for col in cpt.columns if col != 'p']
            toAdd = []
            for i, row in cpt.iterrows():
                toCheck = [row[col] for col in cols]
                for i2, row2 in cpt.iterrows():
                    toCompare = [row2[col] for col in cols]
                    if toCheck == toCompare and i2 != i and (i2, i) not in toAdd:
                        toAdd.append((i, i2))

            for i in range(len(toAdd)):
                cpt.iloc[toAdd[i][0], -1] += cpt.iloc[toAdd[i][1], -1]
            for i in range(len(toAdd)):
                cpt.drop(toAdd[i][1], inplace=True)
            cpt.reset_index(inplace=True, drop=True)
        return cpt

    def maxing_out(self, ordering, cpt):
        """
        :return A cpt with variables in ordering summed out
        """
        checked = []
        for var in ordering:
            #cpt.drop(columns=[var], inplace=True)
            #cols = [col for col in cpt.columns if col != 'p']
            checked.append(var)
            cols = [col for col in cpt.columns if ((col != 'p') and (col not in checked))]
            toAdd = []
            for i, row in cpt.iterrows():
                toCheck = [row[col] for col in cols]
                for i2, row2 in cpt.iterrows():
                    toCompare = [row2[col] for col in cols]
                    if toCheck == toCompare and i2 != i and (i2, i) not in toAdd:
                        toAdd.append((i, i2))
            print(toAdd)
            for i in range(len(toAdd)):
                cpt.iloc[toAdd[i][0], -1] = max(cpt.iloc[toAdd[i][1], -1], cpt.iloc[toAdd[i][0], -1])
            for i in range(len(toAdd)):
                cpt.drop(toAdd[i][1], inplace=True)
            cpt.reset_index(inplace=True, drop=True)
        return cpt


    def create_table(self):
        """
        Creates a single CPT
        :return
        """
        vars = self.bn.get_all_variables()
        z = pd.DataFrame(list(itertools.product(*[self.options for i in range(len(vars))])), columns=vars)
        z['p'] = [1] * z.shape[0]
        return z

    def exclude_evidence(self, cpt, evidence):
        """
        :return Returns the CPT without the rows contradicting the evidence
        """
        if not evidence: return cpt
        else:
            for key in evidence.keys():
                if key in cpt.columns:
                    cpt = cpt[cpt[key] == evidence[key]]
            cpt.reset_index(inplace=True, drop=True)
        return cpt
