from typing import Union
from BayesNet import BayesNet
import sys
from copy import deepcopy
import itertools
import pandas as pd
import random

pd.set_option("display.max_rows", None, "display.max_columns", None)

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
        self.ordering = {"minFill" : self.minFillOrder(), "minDegree": self.minDegreeOrder()}

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
        bn.draw_structure()
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


    def prune_edges(self, e):
        """
        :return A Bayesian Network with pruned edges according to evidence e
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
                bn.update_cpt(children, cpt[cpt[var] == e[var]])
        return bn

    def network_pruning(self, q, e):
        """
        :param q: list of values the query addresses
        :param e: evidence (dict of variables and their values)
        :returns: a pruned bayesian network
        """
        bn = deepcopy(self.bn)
        if type(q) == type(dict()): q = list(q.keys())
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
        for evidence in e.keys():
            del s[evidence]
            pi.remove(evidence)
        for node in q: pi.remove(node)

        for i in range(len(pi)):
            fk = {cp: cpt for cp, cpt in s.items() if pi[i] in cpt.columns}
            f = self.multip_factors(fk.values())
            fi = self.sum_out(f, pi[i])
            for k in fk.keys(): del s[k]
            s['f' + str(i)] = fi
        return s


    def map(self, vars, e={}, ordering="minFill"):
        """
        Maximises the value and instantiation of variables m given evidence e
        :return instantiation and value
        """
        bn = self.network_pruning(vars, e)  # Prune network
        pi = [self.minFillOrder(bn) if ordering == "minFill" else self.minDegreeOrder(bn)][0]  # Elimination order
        pi = [v for v in pi if v not in vars] + vars
        q = bn.get_all_variables()
        s = {var: bn.get_cpt(var) for var in q} # Create a single table based on pi (and possible evidence e)
        for i in range(len(pi)):
            fk = {cp: cpt for cp, cpt in s.items() if pi[i] in cpt.columns}
            f = self.multip_factors(fk.values())
            if pi[i] in vars:
                fi = self.max_out(f, [pi[i]])
            else:
                fi = self.sum_out(f, pi[i])
            for k in fk.keys(): del s[k]
            factor = 'f' + str(i)
            s[factor] = fi
        # Return the most likely instantiation and it's probability
        return s[factor].iloc[s[factor]['p'].idxmax()]


    def mpe(self, e={}, ordering="minFill"):
        """
        Finds the Most Probable Explanation given possible evidence e
        :return instantiation and value
        """
        checked_vars = []
        bn = self.prune_edges(e)
        q = bn.get_all_variables()
        pi = [self.minFillOrder(bn) if ordering == "minFill" else self.minDegreeOrder(bn)][0]
        s = {var: bn.get_cpt(var) for var in q}
        for i in range(len(q)):
            fk = {cp : cpt for cp, cpt in s.items() if pi[i] in cpt.columns}
            f = self.multip_factors(fk.values())
            checked_vars.append(pi[i])
            fi = self.max_out(f, checked_vars)
            for k in fk.keys(): del s[k]
            s['f' + str(i)] = fi
        return s

    def multip_factors(self, cpts):
        """
        Multiplies the corresponding rows of the cpts
        :return z: a cpt containing all the p-values of the cpts
        """
        z = self.union_of_cpts(cpts)
        for cpt in cpts:
            for i, row_content in cpt.iterrows():
                for i_z, row_content_z in z.iterrows():
                    if all([row_content_z[c] == row_content[c] for c in cpt.columns if c != 'p']):
                        z.iloc[i_z, -1] *= row_content['p']
        return z


    def max_out(self, f, checked):
        """
        :return A cpt with variables in ordering summed out
        """
        cols = [col for col in f.columns if (col != 'p') and (col not in checked)]
        toAdd = []
        for i, row in f.iterrows():
            toCheck = [row[col] for col in cols]
            for i2, row2 in f.iterrows():
                toCompare = [row2[col] for col in cols]
                if toCheck == toCompare and i2 != i and (i2, i) not in toAdd:
                    toAdd.append((i, i2))
        toDrop = set()
        for i in range(len(toAdd)):
            if f.iloc[toAdd[i][1], -1] <= f.iloc[toAdd[i][0], -1]: toDrop.add(toAdd[i][1])
            else: toDrop.add(toAdd[i][0])
        for i in toDrop:
            f.drop(i, inplace=True)
        f.reset_index(inplace=True, drop=True)
        return f

    def sum_out(self, f, var):
        """
        :return A cpt with variables in ordering summed out
        """
        f.drop(columns=var, inplace=True)
        cols = [col for col in f.columns if col != 'p']
        toAdd = []
        for i, row in f.iterrows():
            toCheck = [row[col] for col in cols]
            for i2, row2 in f.iterrows():
                toCompare = [row2[col] for col in cols]
                if toCheck == toCompare and i2 != i and (i2, i) not in toAdd:
                    toAdd.append((i, i2))
        for i in range(len(toAdd)):
            f.iloc[toAdd[i][0], -1] += f.iloc[toAdd[i][1], -1]
        for i in range(len(toAdd)):
            f.drop(toAdd[i][1], inplace=True)
        f.reset_index(inplace=True, drop=True)
        return f

    def union_of_cpts(self, cpts):
        """
        :return A union of the cpts given as input
        """
        temp_cpt = [cpt.drop(columns='p') for cpt in cpts]
        temp_cpt = list(temp_cpt)
        z = temp_cpt[0]
        for i in range(1, len(temp_cpt)):
            z = pd.merge(z, temp_cpt[i])
        z['p'] = [1] * z.shape[0]
        z.reset_index(inplace=True, drop=True)
        return z

    def extend_random_network(self, num_nodes=int, num_edges=tuple):
        """
        Extends the already existing bayesian network with variables and edges connecting them to random nodes
        For the purposes of this project the probabilities are always initialized to be 0.5
        :arg num_nodes: The number of nodes the network should be extended to
        :arg num_edges: A tuple containing the minimum and maximum number of edges each node should be connected with
        :returns a randomly generated bayesian network
        """
        for i in range(num_nodes):
            new_node = "node_" + str(i)
            edges = random.sample(self.bn.get_all_variables(), k=random.choice([i for i in range(*num_edges)]))
            cols = edges + [new_node]
            cpt = pd.DataFrame(list(itertools.product(*[self.options for i in range(len(cols))])), columns=cols)
            cpt['p'] = 0.5
            self.bn.add_var(new_node, cpt=cpt)
            for node in edges:
                self.bn.add_edge((node, new_node))
        return self.bn