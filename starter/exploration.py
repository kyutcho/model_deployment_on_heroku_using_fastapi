# %%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv("data/census.csv")
# %%
# Remove leading spaces in columns
new_col = [c.lstrip() for c in df.columns.to_list()]
old_col = df.columns.to_list()

col_name_dict = {}
for o, n in zip(old_col, new_col):
    col_name_dict[o] = n

df = df.rename(col_name_dict, axis=1)
# %%
df.head()
#%%
df.columns.to_list()
#%%
df.dtypes
# %%
df["salary"].value_counts()
#%%
df["race"].unique()[0]
#%%
df["race"].apply(lambda x: x.lstrip())
#%%
cols = df.select_dtypes('object').columns.to_list()

for c in cols:
    df[c] = df[c].apply(lambda x: x.lstrip())
#%%ds
df.loc[df["salary"] == '<=50K', "capital-gain"].mean()
#%%
df.loc[df["salary"] == '>50K', "capital-gain"].mean()
# %%
for cls in df["salary"].unique():
    df.loc[df["salary"] == cls, "capital-gain"]
#%%
d = {"a": 3, "b": 2}
print(d["a"])
# %%
print(pd.DataFrame(np.array([[1,2,3]])))
# %%
import numpy as np
arr = np.array([[0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0]])
# %%
