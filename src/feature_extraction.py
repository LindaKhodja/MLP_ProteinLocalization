import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import ProtParamData
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

def clean_sequence(sequence):
    """Remove invalid characters from the sequence and handle NaN values."""
    if pd.isna(sequence):
        return ''
    if not isinstance(sequence, str):
        return ''
    valid_aa = set('ACDEFGHIKLMNPQRSTVWYZ')
    return ''.join([aa for aa in sequence if aa in valid_aa])

def calculate_gravy(sequence):
    """Calculate the GRAVY (Grand Average of Hydropathy) score."""
    return sum(ProtParamData.kd.get(aa, 0) for aa in sequence) / len(sequence) if sequence else 0

def calculate_aa_composition(sequence):
    """Calculate the amino acid composition of a given sequence."""
    aa_count = Counter(sequence)
    total_length = len(sequence)
    return {aa: count/total_length for aa, count in aa_count.items()}

def extract_terminal_features(sequence, n=30):
    """Extract features from N-terminal and C-terminal regions."""
    n_term = sequence[:n]
    c_term = sequence[-n:]
    return {
        'n_term_hydrophobicity': calculate_gravy(n_term),
        'c_term_hydrophobicity': calculate_gravy(c_term),
        'n_term_charge': sum(1 if aa in 'RK' else (-1 if aa in 'DE' else 0) for aa in n_term),
        'c_term_charge': sum(1 if aa in 'RK' else (-1 if aa in 'DE' else 0) for aa in c_term)
    }

def calculate_hydrophobic_residue_percentage(sequence):
    """Calculate the percentage of hydrophobic residues in the sequence."""
    hydrophobic_residues = set('AVILMFYW')
    hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_residues)
    return hydrophobic_count / len(sequence) if sequence else 0

def calculate_segmented_aa_composition(sequence, n_term_len=60, c_term_len=15):
    """Calculate amino acid composition for N-terminal and C-terminal segments."""
    n_term = sequence[:n_term_len]
    c_term = sequence[-c_term_len:]
    
    n_term_composition = calculate_aa_composition(n_term)
    c_term_composition = calculate_aa_composition(c_term)
    
    # Rename keys to distinguish segment-specific compositions
    n_term_composition = {f'N_Term_AA_Composition_{aa}': freq for aa, freq in n_term_composition.items()}
    c_term_composition = {f'C_Term_AA_Composition_{aa}': freq for aa, freq in c_term_composition.items()}
    
    return {**n_term_composition, **c_term_composition}

def process_sequence(row):
    """Process a single sequence and extract features."""
    sequence = clean_sequence(row['Sequence'])
    
    if not sequence:
        print(f"Warning: Empty or invalid sequence at index {row.name}")
        return None, None  # Return None for both features and label
    
    try:
        analysis = ProteinAnalysis(sequence)
        feature_dict = {
            'Targeting_Peptide_Length': row['Length'],
            'Full_Sequence_Length': len(sequence),
            'Targeting_Peptide_Ratio': row['Length'] / len(sequence) if len(sequence) > 0 else 0,
            'Molecular_Weight': analysis.molecular_weight(),
            'Isoelectric_Point': analysis.isoelectric_point(),
            'GRAVY_Score': calculate_gravy(sequence),
            'Hydrophobic_Residue_Percentage': calculate_hydrophobic_residue_percentage(sequence),
        }
        
        # Global Amino Acid Composition
        feature_dict.update(calculate_aa_composition(sequence))
        
        # Terminal Features
        feature_dict.update(extract_terminal_features(sequence))
        
        # Segmented Amino Acid Composition (N-terminal and C-terminal)
        segmented_aa = calculate_segmented_aa_composition(sequence)
        for k, v in segmented_aa.items():
            feature_dict[k] = v if v is not None else 0
        
        return feature_dict, row['Type']  # Return both features and the valid label
    
    except Exception as e:
        print(f"Error processing sequence at index {row.name}: {e}")
        return None, None  # Return None for both features and label

def extract_features(df):
    """Extract features from the given DataFrame using parallel processing."""
    results = Parallel(n_jobs=-1)(delayed(process_sequence)(row) for _, row in df.iterrows())
    
    features = [result[0] for result in results if result[0] is not None]
    valid_labels = [result[1] for result in results if result[1] is not None]
    
    feature_df = pd.DataFrame(features).fillna(0)
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_columns = feature_df.select_dtypes(include=[np.number]).columns
    feature_df[numerical_columns] = scaler.fit_transform(feature_df[numerical_columns])
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    pca_features = pca.fit_transform(feature_df)
    
    pca_df = pd.DataFrame(pca_features, columns=[f'PCA_{i+1}' for i in range(pca_features.shape[1])])
    
    return pd.concat([feature_df, pca_df], axis=1), valid_labels  # Return features and valid labels

if __name__ == "__main__":
    # Load your data
    data_path = '/mnt/c/Users/Linda/Desktop/GitHub_Projects/MLP_ProteinLocalization/data/processed_data.csv'
    df = pd.read_csv(data_path)
    
    # Extract features and valid labels
    features, labels = extract_features(df)
    
    # Save features to a new CSV file
    feature_output_path = '/mnt/c/Users/Linda/Desktop/GitHub_Projects/MLP_ProteinLocalization/data/features.csv'
    features.to_csv(feature_output_path, index=False)
    
    # Save labels to a separate CSV file
    label_output_path = '/mnt/c/Users/Linda/Desktop/GitHub_Projects/MLP_ProteinLocalization/data/labels.csv'
    pd.Series(labels).to_csv(label_output_path, index=False, header=['Type'])  # Save as DataFrame with header
    
    print(f"Features extracted and saved to {feature_output_path}")
    print(f"Labels saved to {label_output_path}")
    print(f"Total number of features: {features.shape[1]}")
    print(f"Shape of feature matrix: {features.shape}")
    print("\nFirst few rows of the feature matrix:")
    print(features.head())

    # Print information about missing values
    missing_sequences = df['Sequence'].isna().sum()
    print(f"\nNumber of missing sequences: {missing_sequences}")
    print(f"Percentage of missing sequences: {(missing_sequences / len(df)) * 100:.2f}%")