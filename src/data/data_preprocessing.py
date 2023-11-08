import pandas as pd
from pathlib import Path

root_path = Path(__file__).parent.parent.parent

df = pd.read_csv(src_path / "data" / "raw" / "filtered.tsv", sep="\t", index_col='Unnamed: 0')
df = df.drop(columns=['similarity', 'lenght_diff'])
cond = df["trn_tox"] > df["ref_tox"]
df.loc[cond, ["reference", "translation"]] = df.loc[cond, ["translation", "reference"]].values
df.loc[cond, ["ref_tox", "trn_tox"]] = df.loc[cond, ["trn_tox", "ref_tox"]].values
df = df[((df["ref_tox"] > 0.8) & (df["trn_tox"] < 0.1))]
df = df[((df['reference'].str.len() > 30) & (df['translation'].str.len() > 20))]

(src_path / "data" / "interim").mkdir(parents=True, exist_ok=True)

df.to_csv(src_path / "data" / "interim" / "preporcessed.tsv", sep="\t")
