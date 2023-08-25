import time

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.layout import bipartite_layout

user = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

it = 10
w = []
for i in range(it):
    w.append(np.random.randint(1, 10))

E = zip(np.random.choice(user, 10), np.random.choice(items, 10), w)
attributes = pd.DataFrame([{"source": user, "target": items, "weights": weight} for user, items, weight in E])
G = nx.DiGraph()
G.add_nodes_from(user, bipartite=0)
G.add_nodes_from(items, bipartite=1)
G = nx.from_pandas_edgelist(attributes, 'source', 'target', edge_attr='weights')


labels = nx.get_edge_attributes(G, 'weights')
pos = bipartite_layout(G, user)

nx.draw(G, pos, with_labels=True, node_size=1000, node_color='r', edge_color='g', arrowsize=30)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.show()

from itertools import islice

ar = [0] * 5
l = []
for i in user:
    for j in items:
        ans = G.get_edge_data(i, j, default=0)
        if (ans == 0):
            ans1 = 0

        else:
            ans1 = G.get_edge_data(i, j, default=0)['weights']

        l.append(ans1)
length_to_split = [10] * 10
Inputt = iter(l)
Output = [list(islice(Inputt, elem))
          for elem in length_to_split]



from itertools import islice

ar = [0] * 5
l = []
for i in user:
    for j in items:
        ans = G.get_edge_data(i, j, default=0)
        if (ans == 0):
            ans1 = 0

        else:
            ans1 = G.get_edge_data(i, j, default=0)['weights']

        l.append(ans1)
length_to_split = [10] * 10
Inputt = iter(l)
Output = [list(islice(Inputt, elem))
          for elem in length_to_split]




cost = np.array(Output)
from scipy.optimize import linear_sum_assignment

row_ind, col_ind = linear_sum_assignment(cost, True)




from typing import List, Tuple
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_sum_assignment_brute_force(
        cost_matrix: np.ndarray,
        maximize: bool = False) -> Tuple[List[int], List[int]]:
    h = cost_matrix.shape[0]
    w = cost_matrix.shape[1]

    if maximize is True:
        cost_matrix = -cost_matrix

    maximum_cost = float("inf")

    if h >= w:
        for i_idices in itertools.permutations(list(range(h)), min(h, w)):
            row_ind = i_idices
            col_ind = list(range(w))
            cost = cost_matrix[row_ind, col_ind].sum()
            if cost < maximum_cost:
                maximum_cost = cost
                optimal_row_ind = row_ind
                optimal_col_ind = col_ind
    if h < w:
        for j_idices in itertools.permutations(list(range(w)), min(h, w)):
            row_ind = list(range(h))
            col_ind = j_idices
            cost = cost_matrix[row_ind, col_ind].sum()
            if cost < maximum_cost:
                maximum_cost = cost
                optimal_row_ind = row_ind
                optimal_col_ind = col_ind

    return optimal_row_ind, optimal_col_ind


if __name__ == "__main__":
    cost_matrix = np.array(Output)
    start = time.time()
    row_ind, col_ind = linear_sum_assignment_brute_force(
        cost_matrix=cost_matrix, maximize=True)

    maximum_cost = cost_matrix[row_ind, col_ind].sum()
    end = time.time()

    print('The time taken is',end-start)
    print(
        f"\nThe optimal assignment from brute force algorithm is: {list(zip(row_ind, col_ind))}."
    )
    print(f"The maximum cost from brute force algorithm is: {maximum_cost}.")

    start = time.time()

    row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_matrix,
                                             maximize=True)


    maximum_cost = cost_matrix[row_ind, col_ind].sum()
    end=time.time()

    print('The Hungerian Time complexity',start-end)

    print(
        f"The optimal assignment from Hungarian algorithm is: {list(zip(row_ind, col_ind))}."
    )
    print(f"The maximum cost from Hungarian algorithm is: {maximum_cost}.")

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.layout import bipartite_layout

user = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
it = 13
w = []

w = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
E = zip(user, items, w)
attributes = pd.DataFrame([{"source": user, "target": items, "weights": weight} for user, items, weight in E])

G = nx.DiGraph()

G.add_nodes_from(user, bipartite=0)
G.add_nodes_from(items, bipartite=1)

G = nx.from_pandas_edgelist(attributes, 'source', 'target', edge_attr='weights')

labels = nx.get_edge_attributes(G, 'weights')
pos = bipartite_layout(G, user, scale=10)
# plt.figure(figsize =(10,10))
# pos = nx.spring_layout(G,k=0.75)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='r', edge_color='g', arrowsize=30)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.show()


from itertools import islice

ar = [0] * 5
l = []

for i in user:
    for j in items:
        ans = G.get_edge_data(i, j, default=0)
        if (ans == 0):
            ans1 = 0

        else:
            ans1 = G.get_edge_data(i, j, default=0)['weights']

        l.append(ans1)

length_to_split = [10] * 10
Inputt = iter(l)
Output = [list(islice(Inputt, elem))
          for elem in length_to_split]

from typing import List, Tuple
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_sum_assignment_brute_force(
        cost_matrix: np.ndarray,
        maximize: bool = False) -> Tuple[List[int], List[int]]:
    h = cost_matrix.shape[0]
    w = cost_matrix.shape[1]

    if maximize is True:
        cost_matrix = -cost_matrix

    maximum_cost = float("inf")

    if h >= w:
        for i_idices in itertools.permutations(list(range(h)), min(h, w)):
            row_ind = i_idices
            col_ind = list(range(w))
            cost = cost_matrix[row_ind, col_ind].sum()
            if cost < maximum_cost:
                maximum_cost = cost
                optimal_row_ind = row_ind
                optimal_col_ind = col_ind
    if h < w:
        for j_idices in itertools.permutations(list(range(w)), min(h, w)):
            row_ind = list(range(h))
            col_ind = j_idices
            cost = cost_matrix[row_ind, col_ind].sum()
            if cost < maximum_cost:
                maximum_cost = cost
                optimal_row_ind = row_ind
                optimal_col_ind = col_ind

    return optimal_row_ind, optimal_col_ind


if __name__ == "__main__":
    cost_matrix = np.array(Output)

    row_ind, col_ind = linear_sum_assignment_brute_force(
        cost_matrix=cost_matrix, maximize=True)

    maximum_cost = cost_matrix[row_ind, col_ind].sum()

    print(
        f"\nThe optimal assignment from brute force algorithm is: {list(zip(row_ind, col_ind))}."
    )
    print(f"The maximum cost from brute force algorithm is: {maximum_cost}.")

    row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_matrix,
                                             maximize=True)

    maximum_cost = cost_matrix[row_ind, col_ind].sum()

    print(
        f"The optimal assignment from Hungarian algorithm is: {list(zip(row_ind, col_ind))}."
    )
    print(f"The maximum cost from Hungarian algorithm is: {maximum_cost}.")




    
    
    
    



