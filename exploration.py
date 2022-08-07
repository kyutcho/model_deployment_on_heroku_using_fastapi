# #%%
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# # %%
# df = pd.read_csv("starter/data/census.csv")
# # %%
# df.shape
# # %%
# df.describe()
# # %%
# df.describe(include="object")
# # %%
# df.isnull().sum()
# # %%
# sns.distplot(df["age"]);
# # %%
# # sns.histplot(df[" fnlgt"]);
# # %%
# new_col = [c.lstrip() for c in df.columns.to_list()]
# old_col = df.columns.to_list()

# col_name_dict = {}
# for o, n in zip(old_col, new_col):
#     col_name_dict[o] = n

# col_name_dict
# # %%
# df = df.rename(col_name_dict, axis=1)
# # %%
# sns.histplot(df["fnlgt"]);
# # %%
# sns.boxplot(x="education", y="education-num", data=df)
# # %%
# df.columns
# # %%
# df.select_dtypes("object")
# # %%
# df.select_dtypes("object").columns.to_list()
# # %%
# df.select_dtypes("number").columns.to_list()
# # %%
# len(df.columns)
# # %%
# df.info()
# # %%
# df.describe(include="number")
# # %%
# sns.histplot(df["capital-gain"]);
# # %%
# sns.displot(df["capital-loss"]);
# # %%
# alpha=0.000001
# sns.displot(np.log(df["capital-loss"]+alpha));
# # %%

import json
import os

path = os.path.join("starter", "starter", "model_config.json")
with open(path, "r") as f:
    config = json.load(f)

if __name__ == "__main__":
    print(config)