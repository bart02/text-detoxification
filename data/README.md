Download and process the data

Run this in the root directory of the project:
```bash
./src/data/download.sh
mkdir -p data/interim
python src/data/process_data.py
python src/data/to_hf_dataset.py
```