# Protein Subcellular Localization Prediction

This project aims to predict the subcellular localization of proteins from their structure using machine learning techniques, specifically a Multilayer Perceptron (MLP) classifier - (Python Implementation). Accurate prediction of protein localization is crucial for understanding protein function and cellular processes.

## Project Structure
```
MLP_ProteinLocalization/
│
├── data/
│ ├── raw_data.tab
│ ├── processed_data.csv
│ ├── features.csv
│ └── labels.csv
│
├── src/
│ ├── fetch_sequences.py
│ └── feature_extraction.py
│
├── notebooks/
│ ├── data_exploration.ipynb
│ └── mlp_model.ipynb
│
├── results/
│ ├── confusion_matrix_.png (for each application)
│ └── class_distribution.png (before and after SMOTE)
│
├── README.md
└── requirements.txt
```
## Dependencies

This project requires Python 3.7+ and several libraries including pandas, numpy, scikit-learn, and Biopython. For a full list of dependencies, see `requirements.txt`.

## Data

The project uses protein sequence data from https://services.healthtech.dtu.dk/services/TargetP-2.0/swissprot_annotated_proteins.tab. The raw data is stored in `data/raw_data.tab`, and processed data in `data/processed_data.csv`.

## Setup

1. Clone this repository:
git clone https://github.com/LindaKhodja/MLP_ProteinLocalization.git
cd MLP_ProteinLocalization

2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

3. Install the required packages:
pip install -r requirements.txt

## Usage

1. Explore the dataset:
   `notebooks/data_exploration.ipynb` - Provides initial insights into the dataset structure and characteristics.

2. Fetch protein sequences:
python3 src/fetch_sequences.py - This script retrieves protein sequences from UniProt based on the accession numbers in the raw data.

4. Extract features from the sequences:
python src/feature_extraction.py - This script processes the protein sequences and extracts relevant features for machine learning.

4. Train and evaluate the model:
`notebooks/mlp_model.ipynb` - Contains the MLP model implementation, training process, and evaluation metrics.

## Results

The model training script will output performance metrics for each fold of cross-validation, average cross-validation performance, and final test set performance. Confusion matrices and class distribution plots will be saved in the `results/` directory. For a detailed interpretation of the results, please refer to the comments in `notebooks/mlp_model.ipynb`.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
