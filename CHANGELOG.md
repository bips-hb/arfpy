# Changelog

<!--next-version-placeholder-->

## Unreleased

- Fix compatibility with NumPy 2 and pandas 3 (`np.in1d` → `np.isin`, positional Series access via `.iloc`)
- Migrate packaging from `setup.py` to `pyproject.toml` (uv / hatchling)
- Fix the twomoons example notebook: define a held-out `df_test` via `train_test_split` and drop a redundant resampling step

## v0.1.0 (17/04/2023)

- First release of `arfpy`

## v0.1.1 (22/09/2023)

- Update License to MIT
- Allow for variables with zero standard deviation in terminal nodes 
- modify code to lighten dependency requirements: allow for numpy >=1.24 and pandas >= 2.0
