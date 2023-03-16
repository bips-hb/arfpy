
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices

class arf:
  """Adversarial RF (ARF)
  1. fit ARF model with arf()
  2. estimate density with arf.forde()
  3. generate data with arf.forge()
  """
  def __init__(self, x, oob = False, dist = "normal", delta = 0,  max_iters =10, **kwargs):
    x_real = x.copy()
    self.p = x_real.shape[1]
    self.orig_colnames = list(x_real)
    self.dist = dist
    self.oob = oob

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
    clf_0 = RandomForestClassifier( oob_score= 'True', **kwargs) 
    clf_0.fit(x, y)

    iters = 0
    # TO DO -- fix Brier score / accuracy confusion
    acc_0 = clf_0.oob_score_
    print(f'accuracy is {acc_0}')
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
        clf_1 = RandomForestClassifier( oob_score= 'True', **kwargs) 
        clf_1.fit(x, y)

        # update iters and check for convergence
        acc_1 = 1 - clf_1.oob_score_
        print(f"Iteration number {iters} reached accuracy of {acc_1}.")
        iters = iters + 1
        if (acc_1 <= 0.5 + delta or iters >= max_iters):
          converged = True
        else:
          clf_0 = clf_1
    self.clf = clf_0
        
    

  def forde(self):
    """ for density estimation
    """
    # Get terminal nodes for all observations
    pred = self.clf.apply(self.x_real)
    
    self.num_trees = self.clf.n_estimators
    
    # If OOB, use only OOB trees
    if self.oob:
      for tree in range(self.num_trees):
        idx_oob = np.isin(range(self.x_real.shape[0]), _generate_unsampled_indices(self.clf.estimators_[tree].random_state, x.shape[0], x.shape[0]))
        pred[np.invert(idx_oob), tree] = -1
    
    # Get probabilities of terminal nodes for each tree 
    # node_probs dims: [nodeid, tree]
    nbins = np.max(pred)
    def my_bincount(x): 
      res = np.bincount(x[x >= 0], minlength=nbins+1)
      res[res == 1] = 0 # Avoid terminal nodes with just one obs
      return res/np.sum(res)
    self.node_probs = np.apply_along_axis(my_bincount, 0, pred)
    
    # Fit continuous distribution in all terminal nodes
    self.params = pd.DataFrame()
    if np.invert(self.factor_cols).any():
      for tree in range(self.num_trees):
        dt = self.x_real.loc[:, np.invert(self.factor_cols)].copy()
        dt["tree"] = tree
        dt["nodeid"] = pred[:,tree]
        long = pd.melt(dt[dt["nodeid"] >= 0], id_vars = ["tree", "nodeid"])
        if self.dist == "normal":
          res = long.groupby(["tree", "nodeid", "variable"], as_index = False).agg(mean=("value", "mean"), sd=("value", "std"))
        else:
          raise ValueError('Other distributions not yet implemented')
          exit()
        self.params = pd.concat([self.params, res])
    
    # Calculate class probabilities for categorical data in all terminal nodes 
    self.class_probs = pd.DataFrame()
    if self.factor_cols.any():
      for tree in range(self.num_trees):
        dt = self.x_real.loc[:, self.factor_cols].copy()
        dt["tree"] = tree
        dt["nodeid"] = pred[:,tree]
        long = pd.melt(dt[dt["nodeid"] >= 0], id_vars = ["tree", "nodeid"])
        res = long.value_counts(sort = False).rename('freq').reset_index()
        self.class_probs = pd.concat([self.class_probs, res])
    return {"cnt": self.params, "cat": self.class_probs, 
            "forest": self.clf, "meta" : pd.DataFrame(data={"variable": self.orig_colnames, "family": self.dist})}
  # TO DO: think again about the parameters we want to return from density estimation
  
  def forge(self, n):
    """ for data generation
    """
    # Sample new observations and get their terminal nodes
    # nodeids dims: [new obs, tree]
    def myfun(x):
      return np.random.choice(p = x, a = np.shape(self.node_probs)[0], size = n, replace = True)
    nodeids = np.apply_along_axis(myfun, 0, self.node_probs)
    
    # Randomly select tree for each new obs. (mixture distribution with equal prob.)
    sampled_trees = np.random.choice(self.num_trees, size = n)
    sampled_nodes = np.zeros(n, dtype=int)
    for i in range(n):
      sampled_nodes[i] = nodeids[i, sampled_trees[i]]
    sampled_trees_nodes = pd.DataFrame({"obs":range(n), "tree":sampled_trees, "nodeid":sampled_nodes})
    
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
        data_new.loc[:, j] = obs_probs[obs_probs["variable"] == colname].groupby("obs").sample(weights = "freq")["value"].reset_index(drop = True)

      else:
        # Continuous columns: Match estimated distribution parameters with r...() function
        if self.dist == "normal":
          data_new.loc[:, j] = np.random.normal(obs_params.loc[obs_params["variable"] == colname, "mean"], obs_params.loc[obs_params["variable"] == colname, "sd"], size = n)
        else:
          raise ValueError('Other distributions not yet implemented')
    
    # Use original column names
    data_new.set_axis(self.orig_colnames, axis = 1, inplace = True)
    
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


