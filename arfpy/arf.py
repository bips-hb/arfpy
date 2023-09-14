
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
import scipy
from arfpy import utils

class arf:
  """Implements Adversarial Random Forests (ARF) in python
  Usage:
  1. fit ARF model with arf()
  2. estimate density with arf.forde()
  3. generate data with arf.forge().

  :param x: Input data.
  :type x: pandas.Dataframe
  :param num_trees:  Number of trees to grow in each forest, defaults to 30
  :type num_trees: int, optional
  :param delta: Tolerance parameter. Algorithm converges when OOB accuracy is < 0.5 + `delta`, defaults to 0
  :type delta: float, optional
  :param max_iters: Maximum iterations for the adversarial loop, defaults to 10
  :type max_iters: int, optional
  :param early_stop: Terminate loop if performance fails to improve from one round to the next?, defaults to True
  :type early_stop: bool, optional
  :param verbose: Print discriminator accuracy after each round?, defaults to True
  :type verbose: bool, optional
  :param min_node_size: minimum number of samples in terminal node, defaults to 5 
  :type min_node_size: int
  """   
  def __init__(self, x,  num_trees = 30, delta = 0,  max_iters =10, early_stop = True, verbose = True, min_node_size = 5, **kwargs):

    # assertions
    assert isinstance(x, pd.core.frame.DataFrame), f"expected pandas DataFrame as input, got:{type(x)}"
    assert len(set(list(x))) == x.shape[1], f"every column must have a unique column name"
    assert max_iters >= 0, f"negative number of iterations is not allowed: parameter max_iters must be >= 0"
    assert min_node_size > 0, f"minimum number of samples in terminal nodes (parameter min_node_size) must be greater than zero"
    assert num_trees > 0, f"number of trees in the random forest (parameter num_trees) must be greater than zero"
    assert 0 <= delta <= 0.5, f"parameter delta must be in range 0 <= delta <= 0.5"


    # initialize values 
    x_real = x.copy()
    self.p = x_real.shape[1]
    self.orig_colnames = list(x_real)
    self.num_trees = num_trees

    # Find object columns and convert to category
    self.object_cols = x_real.dtypes == "object"
    for col in list(x_real):
      if self.object_cols[col]:
        x_real[col] = x_real[col].astype('category')
    
    # Find factor columns
    self.factor_cols = x_real.dtypes == "category"
    
    # Save factor levels
    self.levels = {}
    for col in list(x_real):
      if self.factor_cols[col]:
        self.levels[col] = x_real[col].cat.categories
    
    # Recode factors to integers
    for col in list(x_real):
      if self.factor_cols[col]:
        x_real[col] = x_real[col].cat.codes
    
    # If no synthetic data provided, sample from marginals
    x_synth = x_real.apply(lambda x: x.sample(frac=1).values)
    
    # Merge real and synthetic data
    x = pd.concat([x_real, x_synth])
    y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])
    # real observations = 0, synthetic observations = 1

    # pass on x_real
    self.x_real = x_real

    # Fit initial RF model
    clf_0 = RandomForestClassifier( oob_score= True, n_estimators=self.num_trees,min_samples_leaf=min_node_size, **kwargs) 
    clf_0.fit(x, y)

    iters = 0

    acc_0 = clf_0.oob_score_ # is accuracy directly
    acc = [acc_0]

    if verbose is True:
      print(f'Initial accuracy is {acc_0}')

    if (acc_0 > 0.5 + delta and iters < max_iters):
      converged = False
      while (not converged): # Start adversarial loop
        # get nodeIDs
        nodeIDs = clf_0.apply(self.x_real) # dimension [terminalnode, tree]

        # add observation ID to x_real
        x_real_obs = x_real.copy()
        x_real_obs['obs'] = range(0,x_real.shape[0])

        # add observation ID to nodeIDs
        nodeIDs_pd = pd.DataFrame(nodeIDs)
        tmp = nodeIDs_pd.copy()
        #tmp.columns = [ "tree" + str(c) for c in tmp.columns ]
        tmp['obs'] = range(0,x_real.shape[0])
        tmp = tmp.melt(id_vars=['obs'], value_name="leaf", var_name="tree")

        # match real data to trees and leafs (node id for tree)
        x_real_obs = pd.merge(left=x_real_obs, right=tmp, on=['obs'], sort=False)
        x_real_obs.drop('obs', axis = 1, inplace= True)

        # sample leafs
        tmp.drop("obs", axis=1, inplace=True)
        tmp = tmp.sample(x_real.shape[0], axis=0, replace=True)
        tmp = pd.Series(tmp.value_counts(sort = False ), name = 'cnt').reset_index()
        draw_from = pd.merge(left = tmp, right = x_real_obs, on=['tree', 'leaf'], sort=False )

        # sample synthetic data from leaf
        grpd =  draw_from.groupby(['tree', 'leaf'])
        x_synth = [grpd.get_group(ind).apply(lambda x: x.sample(n=grpd.get_group(ind)['cnt'].iloc[0], replace = True).values) for ind in grpd.indices]
        x_synth = pd.concat(x_synth).drop(['cnt', 'tree', 'leaf'], axis=1)
        
        # delete unnecessary objects 
        del(nodeIDs, nodeIDs_pd, tmp, x_real_obs, draw_from)

        # merge real and synthetic data
        x = pd.concat([x_real, x_synth])
        y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])
        
        # discrimintator
        clf_1 = RandomForestClassifier( oob_score= True, n_estimators=self.num_trees, min_samples_leaf=min_node_size,**kwargs) 
        clf_1.fit(x, y)

        # update iters and check for convergence
        acc_1 = clf_1.oob_score_
        
        acc.append(acc_1)
        
        iters = iters + 1
        plateau = True if early_stop is True and acc[iters] > acc[iters - 1] else False
        if verbose is True:
          print(f"Iteration number {iters} reached accuracy of {acc_1}.")
        if (acc_1 <= 0.5 + delta or iters >= max_iters or plateau):
          converged = True
        else:
          clf_0 = clf_1
    self.clf = clf_0
    self.acc = acc 
        
    # Pruning
    pred = self.clf.apply(self.x_real)
    for tree_num in range(0, self.num_trees):
      tree = self.clf.estimators_[tree_num]
      left = tree.tree_.children_left
      right = tree.tree_.children_right
      leaves = np.where(left < 0)[0]

      # get leaves that are too small
      unique, counts = np.unique(pred[:, tree_num], return_counts=True)
      to_prune = unique[counts < min_node_size]

      # also add leaves with 0 obs.
      to_prune = np.concatenate([to_prune, np.setdiff1d(leaves, unique)])

      while len(to_prune) > 0:
        for tp in to_prune:
          # Find parent
          parent = np.where(left == tp)[0]
          if len(parent) > 0:
            # Left child
            left[parent] = right[parent]
          else:
            # Right child
            parent = np.where(right == tp)[0]
            right[parent] = left[parent]
        # Prune again if child was pruned
        to_prune = np.where(np.in1d(left, to_prune))[0]

  def forde(self, dist = "truncnorm", oob = False, alpha = 0):
    """This part is for density estimation (FORDE)

    :param dist: Distribution to use for density estimation of continuous features. Distributions implemented so far: "truncnorm", defaults to "truncnorm"
    :type dist: str, optional
    :param oob: Only use out-of-bag samples for parameter estimation? If `True`, `x` must be the same dataset used to train `arf`, defaults to False
    :type oob: bool, optional
    :param alpha: Optional pseudocount for Laplace smoothing of categorical features. This avoids zero-mass points when test data fall outside the support of training data. Effectively parametrizes a flat Dirichlet prior on multinomial likelihoods, defaults to 0
    :type alpha: float, optional
    :return: Return parameters for the estimated density.
    :rtype: dict
    """    
 
    self.dist = dist
    self.oob = oob
    self.alpha = alpha

    # Get terminal nodes for all observations
    pred = self.clf.apply(self.x_real)
    
    # If OOB, use only OOB trees
    if self.oob:
      for tree in range(self.num_trees):
        idx_oob = np.isin(range(self.x_real.shape[0]), _generate_unsampled_indices(self.clf.estimators_[tree].random_state, self.x.shape[0], self.x.shape[0]))
        pred[np.invert(idx_oob), tree] = -1
    
    # Get probabilities of terminal nodes for each tree 
    # node_probs dims: [nodeid, tree]
    #self.node_probs = np.apply_along_axis(func1d= utils.bincount, axis = 0, arr =pred, nbins = np.max(pred))
    
    # compute leaf bounds and coverage
    bnds = pd.concat([utils.bnd_fun(tree=j, p = self.p, forest = self.clf, feature_names = self.orig_colnames) for j in range(self.num_trees)])
    bnds['f_idx']= bnds.groupby(['tree', 'leaf']).ngroup()

    bnds_2 = pd.DataFrame()
    for t in range(self.num_trees):
      unique, freq = np.unique(pred[:,t], return_counts=True)
      vv = pd.concat([pd.Series(unique, name = 'leaf'), pd.Series(freq/pred.shape[0], name = 'cvg')], axis = 1)
      zz = bnds[bnds['tree'] == t]
      bnds_2 =pd.concat([bnds_2,pd.merge(left=vv, right=zz, on=['leaf'])])
    bnds = bnds_2
    del(bnds_2)

    # set coverage for nodes with single observations to zero
    if np.invert(self.factor_cols).any():
      bnds.loc[bnds['cvg'] == 1/pred.shape[0],'cvg'] = 0
    
    # no parameters to learn for zero coverage leaves - drop zero coverage nodes
    bnds = bnds[bnds['cvg'] > 0]

    # rename leafs to nodeids
    bnds.rename(columns={'leaf': 'nodeid'}, inplace=True)

    # save bounds to later use coverage for drawing new samples
    self.bnds= bnds
    # Fit continuous distribution in all terminal nodes
    self.params = pd.DataFrame()
    if np.invert(self.factor_cols).any():
      for tree in range(self.num_trees):
        dt = self.x_real.loc[:, np.invert(self.factor_cols)].copy()
        dt["tree"] = tree
        dt["nodeid"] = pred[:,tree]
        # merge bounds and make it long format
        long = pd.merge(right = bnds[['tree', 'nodeid','variable', 'min', 'max', 'f_idx']], left = pd.melt(dt[dt["nodeid"] >= 0], id_vars = ["tree", "nodeid"]), on = ['tree', 'nodeid', 'variable'], how = 'left')
        # get distribution parameters
        if self.dist == "truncnorm":
          res = long.groupby([ 'tree',"nodeid", "variable"], as_index = False).agg(mean=("value", "mean"), sd=("value", "std"), min = ("min", "min"), max = ("max", "max"))
        else:
          raise ValueError('unknown distribution, make sure to enter a vaild value for dist')
          exit()
        self.params = pd.concat([self.params, res])
    
    # Get class probabilities in all terminal nodes
    self.class_probs = pd.DataFrame()
    if self.factor_cols.any():
      for tree in range(self.num_trees):
        dt = self.x_real.loc[:, self.factor_cols].copy()
        dt["tree"] = tree
        dt["nodeid"] = pred[:,tree]
        dt = pd.melt(dt[dt["nodeid"] >= 0], id_vars = ["tree", "nodeid"])
        long = pd.merge(left = dt, right = bnds, on = ['tree','nodeid', 'variable'])
        long['count_var'] = long.groupby(['tree', 'nodeid', 'variable'])['variable'].transform('count')
        long['count_var_val'] = long.groupby(['tree', 'nodeid', 'variable', 'value'])['variable'].transform('count')
        long.drop_duplicates(inplace=True)
        if self.alpha == 0:
          long['prob'] = long['count_var_val'] / long['count_var'] 
        else:
          # Define the range of each variable in each leaf
          long['k'] = long.groupby(['variable'])['value'].transform('nunique')  
          long.loc[long['min'] == float('-inf') , 'min'] = 0.5 - 1
          long.loc[long['max'] == float('inf') , 'max'] = long['k'] + 0.5 - 1
          long.loc[round(long['min'] % 1,2) != 0.5 , 'min'] = long['min'] - 0.5
          long.loc[round(long['max'] % 1,2) != 0.5 , 'min'] = long['max'] + 0.5
          long['k'] = long['max'] - long['min']  
          # Enumerate each possible leaf-variable-value combo
          tmp = long[['f_idx','tree', "nodeid", 'variable', 'min','max']].copy()
          tmp['rep_min'] = tmp['min'] + 0.5 
          tmp['rep_max'] = tmp['max'] - 0.5 
          tmp['levels'] = tmp.apply(lambda row:  list(range(int(row['rep_min']), int(row['rep_max'] + 1))), axis=1)
          tmp = tmp.explode('levels')
          cat_val = pd.DataFrame(self.levels).melt()
          cat_val['levels'] = cat_val['value'] 
          tmp =  pd.merge(left = tmp, right = cat_val, on = ['variable', 'levels'])[['variable', 'f_idx','tree', "nodeid",'value']]
          # populate count, k
          tmp = pd.merge(left = tmp, right = long[['f_idx', 'variable', 'tree', "nodeid",'count_var', 'k']], on = ['f_idx', "nodeid", 'variable', 'tree'])
          # Merge with long, set val_count = 0 for possible but unobserved levels
          long = pd.merge(left = tmp, right = long, on = ['f_idx','tree',"nodeid",  'variable','value','count_var','k'], how = 'left')
          long.loc[long['count_var_val'].isna(), 'count_var_val'] = 0
          long = long[['f_idx','tree',"nodeid",  'variable', 'value', 'count_var_val', 'count_var', 'k']].drop_duplicates()
          # Compute posterior probabilities
          long['prob'] = (long['count_var_val'] + self.alpha) / (long['count_var'] + self.alpha*long['k'])
          long['value'] = long['value'].astype('int8')
        
        long = long[['f_idx','tree', "nodeid", 'variable', 'value','prob']]
        self.class_probs = pd.concat([self.class_probs, long])
    return {"cnt": self.params, "cat": self.class_probs, 
            "forest": self.clf, "meta" : pd.DataFrame(data={"variable": self.orig_colnames, "family": self.dist})}
  # TO DO: optional -- think of dropping f_idx
  def forge(self, n):
    """This part is for data generation (FORGE)

    :param n: Number of synthetic samples to generate.
    :type n: int
    :return: Returns generated data.
    :rtype: pandas.DataFrame
    """
    try:
      getattr(self, 'bnds')
    except AttributeError:
      raise AttributeError('need density estimates to generate data -- run .forde() first!')

    # Sample new observations and get their terminal nodes
    # Draw random leaves with probability proportional to coverage
    unique_bnds = self.bnds[['tree', 'nodeid', 'cvg']].drop_duplicates()
    draws = np.random.choice(a=range(unique_bnds.shape[0]), p = unique_bnds['cvg'] / self.num_trees, size=n)
    sampled_trees_nodes = unique_bnds[['tree','nodeid']].iloc[draws,].reset_index(drop =True).reset_index().rename(columns={'index': 'obs'})

    # Get distributions parameters for each new obs.
    if np.invert(self.factor_cols).any():
      obs_params = pd.merge(sampled_trees_nodes, self.params, on = ["tree", "nodeid"]).sort_values(by=['obs'], ignore_index = True)
    
    # Get probabilities for each new obs.
    if self.factor_cols.any():
      obs_probs = pd.merge(sampled_trees_nodes, self.class_probs, on = ["tree", "nodeid"]).sort_values(by=['obs'], ignore_index = True)
    
    # Sample new data from mixture distribution over trees
    data_new = pd.DataFrame(index=range(n), columns=range(self.p))
    for j in range(self.p): 
      colname = self.orig_colnames[j]
      
      if self.factor_cols[j]:
        # Factor columns: Multinomial distribution
        data_new.isetitem(j, obs_probs[obs_probs["variable"] == colname].groupby("obs").sample(weights = "prob")["value"].reset_index(drop = True))

      else:
        # Continuous columns: Match estimated distribution parameters with r...() function
        if self.dist == "truncnorm":
         # sample from normal distribution, only here for debugging
         # data_new.loc[:, j] = np.random.normal(obs_params.loc[obs_params["variable"] == colname, "mean"], obs_params.loc[obs_params["variable"] == colname, "sd"], size = n) 
         
         # sample from truncated normal distribution
         # note: if sd == 0, truncnorm will return location parameter -> this is desired; if we have 
         # all obs. in that leave having the same value, we sample a new obs. with exactly that value as well
         myclip_a = obs_params.loc[obs_params["variable"] == colname, "min"]
         myclip_b = obs_params.loc[obs_params["variable"] == colname, "max"]
         myloc = obs_params.loc[obs_params["variable"] == colname, "mean"]
         myscale = obs_params.loc[obs_params["variable"] == colname, "sd"]
         data_new.isetitem(j, scipy.stats.truncnorm(a =(myclip_a - myloc) / myscale,b = (myclip_b - myloc) / myscale, loc = myloc , scale = myscale ).rvs(size = n))
         del(myclip_a,myclip_b,myloc,myscale)
        else:
          raise ValueError('Other distributions not yet implemented')
    
    # Use original column names
    data_new = data_new.set_axis(self.orig_colnames, axis = 1, copy = False)
    
    # Convert categories back to category   
    for col in self.orig_colnames:
      if self.factor_cols[col]:
        data_new[col] = pd.Categorical.from_codes(data_new[col], categories = self.levels[col])

    # Convert object columns back to object
    for col in self.orig_colnames:
      if self.object_cols[col]:
        data_new[col] = data_new[col].astype("object")

    # Return newly sampled data
    return data_new


