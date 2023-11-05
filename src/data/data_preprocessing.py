import pandas as pd

df = pd.read_csv("../../data/raw/filtered.tsv", sep="\t", index_col='Unnamed: 0')
df = df.drop(columns=['similarity', 'lenght_diff'])
cond = df["trn_tox"] > df["ref_tox"]
df.loc[cond, ["reference", "translation"]] = df.loc[cond, ["translation", "reference"]].values
df.loc[cond, ["ref_tox", "trn_tox"]] = df.loc[cond, ["trn_tox", "ref_tox"]].values
df = df[((df["ref_tox"] > 0.8) & (df["trn_tox"] < 0.1))]
df = df[((df['reference'].str.len() > 30) & (df['translation'].str.len() > 20))]

df.to_csv("../../data/interim/preporcessed.tsv", sep="\t")
