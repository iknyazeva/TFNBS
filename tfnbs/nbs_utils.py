import numpy as np
import numpy.typing as npt
from tfnbs.utils import get_components

def nbs_bct(group1, group2, threshold, n_permutations = 100, tail = 'both', 
            paired = False, random_seed = 42)

    ix, iy, iz = group1.shape
    
    # Initial T Test for the groups and acquire T-statistics matrix 
    if paired == False:
        t_stat = compute_t_stat(taskA, taskB, paired = False)
        # Make seperate instances for when taska & task b are different diensions ix2, iy2, iz2 = shape.g1/g2
    else:
        t_stat = compute_t_stat(taskA, taskB, paired = True)
        #t_stat_diff = compute_t_stat_diff(taskA, taskB, paired = True)
    

    # Get binary matrix - w.r.t threshold 
    t_stats_g1 = t_stat['g2>g1']
    t_stats_g2 = t_stat['g1>g2']

    # Extract indices of corr greater than threshold 
    ind_t_g1 = np.where(t_stats_g1[0] > threshold)
    ind_t_g2 = np.where(t_stats_g2[0] > threshold)
    np.fill_diagonal([ind_t_g1, ind_t_g2], 0)
    

    # intialize empty matrix of size
    adj = np.zeros((100, 100))
    a, sz = get_components(ind_t_g1)
    uni_labels = np.unique(a)

    # ---------
    # to acqure observed component siez in number of edges
    observed_sizes = []
    for label in component_labels:
        if sz[label - 1] > 1:
            nodes = np.where(a == label)[0]
            subgraph = adj[np.ix_(nodes, nodes)]
            size = np.sum(subgraph) / 2  # undirected
            observed_sizes.append(size)
    # rework --?

    # Randomize labels -> Peform tt test - adjmatrix - find connected components -> accumulate connected componbets in varaible
    


    # Calculate p value for number of cnnected edges/
    pvals = [(np.sum(np.array(null_dist) >= size) + 1) / (n_permutations + 1) for size in observed_sizes] # internally 


    return p_vals, adj, null



# inputs: Group 1, Group2, threshold, permutation, paired, seed, 
# myabe tail? 


# Initial T Test for the groups and acquire T-statistics matrix 

# Get binary matrix - w.r.t threshold 

# Get adjacency matrix
# Calculate conected components using get_Components

# Randomize labels and then perform ttest - adj matrix - find connected components -> accumulate connectd components in variable 


# Calcuilate p value: 
# Pvalues - for number connected edges 
# 