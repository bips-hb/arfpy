# put helper functions in this file

import numpy as np 
import pandas as pd

def bnd_fun(tree, p, forest, feature_names ):
    my_tree = forest.estimators_[tree].tree_
    num_nodes = my_tree.node_count
    lb = np.full(shape=(num_nodes, p), fill_value=float('-inf'))
    ub = np.full(shape=(num_nodes, p), fill_value=float('inf'))
    for i in range(num_nodes):
        left_child = my_tree.children_left[i]
        right_child = my_tree.children_right[i]
        if left_child > -1: # leaf nodes are indicated by -1
            ub[left_child,: ] = ub[right_child, :] = ub[i,: ]
            lb[left_child,: ] = lb[right_child,: ] = lb[i,: ]
            if left_child != right_child:
                # If no pruned node, split changes bounds
                ub[left_child, my_tree.feature[i]] = lb[right_child,  my_tree.feature[i]] = my_tree.threshold[i]
    leaves = np.where(my_tree.children_left < 0)[0]
    # lower and upper bounds, to long format, to single data frame for return
    l = pd.concat([pd.Series(np.full(shape=(leaves.shape[0]), fill_value=tree), name = 'tree'), 
                             pd.Series(leaves, name = 'leaf'), 
                             pd.DataFrame(lb[leaves,], columns = feature_names)], axis = 1)
    u = pd.concat([pd.Series(np.full(shape=(leaves.shape[0]), fill_value=tree), name = 'tree'), 
                             pd.Series(leaves, name = 'leaf'), 
                             pd.DataFrame(ub[leaves,], columns = feature_names)], axis = 1)
    ret = pd.merge(left=pd.melt(l, id_vars=['tree', 'leaf'], value_name='min'),
              right = pd.melt(u, id_vars=['tree', 'leaf'], value_name='max'),
              on=['tree','leaf', 'variable'])
    del(l,u)
    return ret
    