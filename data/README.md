Download and process the data

Run this in the root directory of the project:
```bash
./src/data/download.sh
python src/data/data_preprocessing.py
python src/data/to_hf_datasets.py
```