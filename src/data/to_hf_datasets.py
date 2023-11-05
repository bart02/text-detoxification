import pandas as pd
from datasets import Dataset, DatasetDict

df = pd.read_csv("../../data/interim/preporcessed.tsv", sep="\t", index_col='Unnamed: 0')
dataset = Dataset.from_pandas(df.drop(columns=['ref_tox', 'trn_tox']), preserve_index=False)

train_test_split = dataset.train_test_split(test_size=0.05)
validation_test_split = train_test_split['test'].train_test_split(test_size=0.8)
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': validation_test_split['train'],
    'test': validation_test_split['test']
})

dataset_dict.save_to_disk("../../data/interim/splitted_dataset")