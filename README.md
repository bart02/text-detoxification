# PMLDL Course, Assignment 1, Text Detoxification
Batalov Artem, a.batalov@innopolis.university, BS20-AI-01

Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style. In this assignment, we are given a dataset of pairs of toxic and non-toxic texts. The task is to create a model that would transform toxic text into non-toxic one.  
[Full task description](task_description.md)

## Solution
[Full report](reports/final_solution.md)

TL;DR: I used few-shot Mistral model and fine-tuned T5 model on given dataset. Fine-tuned T5 model showed better results (0.45 score, see Evaluation section in the report) than Mistral model (0.2 score).

## How to use
1. Clone this repository
   ```bash
   git clone https://github.com/bart02/text-detoxification.git
   cd text-detoxification
   ```
2. Install requirements
   ```bash
   pip install -r requirements.txt
   ```
3. Download and process the data
   ```bash
   ./src/data/download.sh
   python src/data/data_preprocessing.py
   python src/data/to_hf_datasets.py
   ```
4. Train the model
   ```bash
   python src/models/train_t5.py
   ```
5. Make predictions
   ```bash
   python src/models/predict_t5.py
   ```
