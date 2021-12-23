from BNReasoner import BNReasoner
import pandas as pd
import random
import time
from BayesNet import BayesNet
from matplotlib import pyplot as plt

reasoner = BNReasoner('testing\covid.BIFXML')

min_networkSize = 90
max_networkSize = 140

print("Staring measurements")

measurements_map = pd.DataFrame([], columns=['Ordering', 'NetworkSize', 'VarSize', 'EvidenceSize', 'TimeTaken'])
for order in ["minFill", "minDegree"]:
    for net_size in range(min_networkSize, max_networkSize, 10):
        reasoner = BNReasoner('testing\covid.BIFXML')
        reasoner.extend_random_network(net_size, (1, 3))
        all_variables = reasoner.bn.get_all_variables()
        print(len(all_variables))
        for q in range(6):
            q_vars = max(q * 10, 1)
            print("\t{} Variables".format(q_vars))
            for e in range(6):
                e_vars = max(e * 10, 1)
                print("\t\t{} evidence".format(e_vars))
                evidence_vars = random.choices(all_variables, k=e_vars)
                evidence = {v: random.choice([False, True]) for v in evidence_vars}
                query_var = random.choices(list(set(all_variables) - set(evidence_vars)), k=q_vars)
                start = time.time()
                reasoner.map(query_var, evidence, ordering=order)
                end = time.time()
                timeTaken = end - start
                measurements_map.loc[len(measurements_map.index)] = [order, len(all_variables), e_vars, q_vars, timeTaken]
measurements_map.to_csv("map_measurements.csv")

measurements_mpe = pd.DataFrame([], columns=['Ordering', 'NetworkSize', 'EvidenceSize', 'TimeTaken'])
for order in ["minFill", "minDegree"]:
    for net_size in range(min_networkSize, max_networkSize, 100):
        reasoner = BNReasoner('testing\covid.BIFXML')
        reasoner.extend_random_network(net_size, (1, 3))
        all_variables = reasoner.bn.get_all_variables()
        print(len(all_variables))
        for e in range(1, 5):
            e_vars = max(e * 10, 1)
            print("\t\t{} evidence".format(e_vars))
            evidence_vars = random.choices(all_variables, k=e_vars)
            evidence = {v: random.choice([False, True]) for v in evidence_vars}
            start = time.time()
            reasoner.mpe(evidence, ordering=order)
            end = time.time()
            timeTaken = end - start
            measurements_mpe.loc[len(measurements_mpe.index)] = [order, net_size, e_vars, timeTaken]
measurements_mpe.to_csv("mpe_measurements.csv")