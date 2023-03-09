
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices

class genrf:
  """Generative RF"""
  def __init__(self, x, oob = False, dist = "normal", **kwargs):
    x_real = x.copy()
    self.p = x_real.shape[1]
    self.orig_colnames = list(x_real)
    self.dist = dist
    
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
    
    # Fit RF to both data
    clf = RandomForestClassifier(**kwargs) 
    clf.fit(x, y)
    
    # Get terminal nodes for all observations
    pred = clf.apply(x_real)
    
    self.num_trees = clf.n_estimators
    
    # If OOB, use only OOB trees
    if oob:
      for tree in range(self.num_trees):
        idx_oob = np.isin(range(x_real.shape[0]), _generate_unsampled_indices(clf.estimators_[tree].random_state, x.shape[0], x.shape[0]))
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
        dt = x_real.loc[:, np.invert(self.factor_cols)].copy()
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
        dt = x_real.loc[:, self.factor_cols].copy()
        dt["tree"] = tree
        dt["nodeid"] = pred[:,tree]
        long = pd.melt(dt[dt["nodeid"] >= 0], id_vars = ["tree", "nodeid"])
        res = long.value_counts(sort = False).rename('freq').reset_index()
        self.class_probs = pd.concat([self.class_probs, res])
  
  def sample(self, n):
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


