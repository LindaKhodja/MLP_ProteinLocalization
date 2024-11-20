import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import ProtParamData
from collections import Counter
from joblib import Parallel, delayed
import logging
import argparse
import os

PROJECT_ROOT = '/mnt/c/Users/Linda/Desktop/GitHub_Projects/MLP_ProteinLocalization'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_sequence(sequence):
    if pd.isna(sequence) or not isinstance(sequence, str):
        return ''
    valid_aa = set('ACDEFGHIKLMNPQRSTVWYZ')
    return ''.join([aa.upper() for aa in sequence if aa.upper() in valid_aa])

def calculate_gravy(sequence):
    return sum(ProtParamData.kd.get(aa, 0) for aa in sequence) / len(sequence) if sequence else 0

def calculate_aa_composition(sequence):
    aa_count = Counter(sequence)
    total_length = len(sequence)
    return {aa: count/total_length for aa, count in aa_count.items()}

def extract_terminal_features(sequence, n=30):
    n_term = sequence[:n]
    c_term = sequence[-n:]
    return {
        'n_term_hydrophobicity': calculate_gravy(n_term),
        'c_term_hydrophobicity': calculate_gravy(c_term),
        'n_term_charge': sum(1 if aa in 'RK' else (-1 if aa in 'DE' else 0) for aa in n_term),
        'c_term_charge': sum(1 if aa in 'RK' else (-1 if aa in 'DE' else 0) for aa in c_term)
    }

def calculate_hydrophobic_residue_percentage(sequence):
    hydrophobic_residues = set('AVILMFYW')
    hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_residues)
    return hydrophobic_count / len(sequence) if sequence else 0

def calculate_segmented_aa_composition(sequence, n_term_len=60, c_term_len=15):
    n_term = sequence[:n_term_len]
    c_term = sequence[-c_term_len:]
    
    n_term_composition = calculate_aa_composition(n_term)
    c_term_composition = calculate_aa_composition(c_term)
    
    n_term_composition = {f'N_Term_AA_Composition_{aa}': freq for aa, freq in n_term_composition.items()}
    c_term_composition = {f'C_Term_AA_Composition_{aa}': freq for aa, freq in c_term_composition.items()}
    
    return {**n_term_composition, **c_term_composition}

def process_sequence(row):
    sequence = clean_sequence(row['Sequence'])
    
    if not sequence:
        logging.warning(f"Empty or invalid sequence at index {row.name}")
        return None, None
    
    try:
        analysis = ProteinAnalysis(sequence)
        feature_dict = {
            'Full_Sequence_Length': len(sequence),
            'Molecular_Weight': analysis.molecular_weight(),
            'Isoelectric_Point': analysis.isoelectric_point(),
            'GRAVY_Score': calculate_gravy(sequence),
            'Hydrophobic_Residue_Percentage': calculate_hydrophobic_residue_percentage(sequence),
        }
        
        feature_dict.update(calculate_aa_composition(sequence))
        feature_dict.update(extract_terminal_features(sequence))
        
        segmented_aa = calculate_segmented_aa_composition(sequence)
        for k, v in segmented_aa.items():
            feature_dict[k] = v if v is not None else 0
        
        secondary_structure = analysis.secondary_structure_fraction()
        feature_dict.update({
            'Helix_Fraction': secondary_structure[0],
            'Turn_Fraction': secondary_structure[1],
            'Sheet_Fraction': secondary_structure[2]
        })
        
        return feature_dict, row['Type']
    
    except Exception as e:
        logging.error(f"Error processing sequence at index {row.name}: {e}")
        return None, None

def extract_features(df):
    results = Parallel(n_jobs=-1)(delayed(process_sequence)(row) for _, row in df.iterrows())
    
    features = [result[0] for result in results if result[0] is not None]
    valid_labels = [result[1] for result in results if result[1] is not None]
    
    feature_df = pd.DataFrame(features).fillna(0)
    
    return feature_df, valid_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from protein sequences.')
    parser.add_argument('--input', default=os.path.join(PROJECT_ROOT, 'data', 'processed_data.csv'), help='Path to input CSV file')
    parser.add_argument('--output_features', default=os.path.join(PROJECT_ROOT, 'data', 'features.csv'), help='Path to output features CSV file')
    parser.add_argument('--output_labels', default=os.path.join(PROJECT_ROOT, 'data', 'labels.csv'), help='Path to output labels CSV file')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    
    features, labels = extract_features(df)
    
    features.to_csv(args.output_features, index=False)
    pd.Series(labels).to_csv(args.output_labels, index=False, header=['Type'])
    
    logging.info(f"Features extracted and saved to {args.output_features}")
    logging.info(f"Labels saved to {args.output_labels}")
    logging.info(f"Total number of features: {features.shape[1]}")
    logging.info(f"Shape of feature matrix: {features.shape}")
    logging.info(f"Feature names: {features.columns.tolist()}")
    
    missing_sequences = df['Sequence'].isna().sum()
    logging.info(f"Number of missing sequences: {missing_sequences}")
    logging.info(f"Percentage of missing sequences: {(missing_sequences / len(df)) * 100:.2f}%")