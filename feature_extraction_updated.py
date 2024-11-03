import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import ProtParamData
from collections import Counter

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
    return {f'Amino_Acid_Composition_{aa}': aa_count.get(aa, 0) / total_length for aa in 'ACDEFGHIKLMNPQRSTVWY'}

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


def extract_features(df):
    """Extract features from the given DataFrame with additional enhancements."""
    features = []
    
    for index, row in df.iterrows():
        sequence = clean_sequence(row['Sequence'])
        
        # Skip row if sequence is empty or invalid
        if not sequence:
            print(f"Warning: Empty or invalid sequence at index {index}")
            continue
        
        try:
            analysis = ProteinAnalysis(sequence)
            feature_dict = {
                'Targeting_Peptide_Length': row['Length'],
                'Full_Sequence_Length': len(sequence),
                'Normalized_Peptide_Length': row['Length'] / len(sequence) if len(sequence) > 0 else 0,
                'Molecular_Weight': analysis.molecular_weight(),
                'Isoelectric_Point': analysis.isoelectric_point(),
                'GRAVY_Score': calculate_gravy(sequence),
                'Hydrophobic_Residue_Percentage': calculate_hydrophobic_residue_percentage(sequence),
            }
            
            # Global Amino Acid Composition
            aa_composition = calculate_aa_composition(sequence)
            for aa, freq in aa_composition.items():
                feature_dict[aa] = freq if freq is not None else 0
            
            # Terminal Features
            feature_dict.update(extract_terminal_features(sequence))
            
            # Segmented Amino Acid Composition (N-terminal and C-terminal)
            segmented_aa = calculate_segmented_aa_composition(sequence)
            for k, v in segmented_aa.items():
                feature_dict[k] = v if v is not None else 0
            
            # Binary flags for subcellular location types
            feature_dict.update({
                'Is_SP': int(row['Type'] == 'SP'),
                'Is_MT': int(row['Type'] == 'MT'),
                'Is_CH': int(row['Type'] == 'CH'),
                'Is_TH': int(row['Type'] == 'TH'),
                'Is_Other': int(row['Type'] == 'Other'),
            })
            
            features.append(feature_dict)
        
        except Exception as e:
            print(f"Error processing sequence at index {index}: {e}")
    
    # Convert feature list to DataFrame and replace any remaining NaN with 0
    feature_df = pd.DataFrame(features).fillna(0)
    return feature_df

    """Extract features from the given DataFrame."""
    features = []
    
    for index, row in df.iterrows():
        sequence = clean_sequence(row['Sequence'])
        
        # Skip row if sequence is empty or invalid
        if not sequence:
            print(f"Warning: Empty or invalid sequence at index {index}")
            continue
        
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
            aa_composition = calculate_aa_composition(sequence)
            for aa, freq in aa_composition.items():
                feature_dict[aa] = freq if freq is not None else 0
            
            # Terminal Features
            feature_dict.update(extract_terminal_features(sequence))
            
            # Segmented Amino Acid Composition (N-terminal and C-terminal)
            segmented_aa = calculate_segmented_aa_composition(sequence)
            for k, v in segmented_aa.items():
                feature_dict[k] = v if v is not None else 0
            
            features.append(feature_dict)
        
        except Exception as e:
            print(f"Error processing sequence at index {index}: {e}")
    
    # Convert feature list to DataFrame and replace any remaining NaN with 0
    feature_df = pd.DataFrame(features).fillna(0)
    return feature_df

if __name__ == "__main__":
    # Load your data
    data_path = '/mnt/c/Users/marwa/OneDrive/Desktop/ird/ClassifHackathon//processed_data.csv'  
    df = pd.read_csv(data_path)
    
    # Extract features
    features = extract_features(df)
    
    # Save features to a new CSV file
    output_path = '/mnt/c/Users/marwa/OneDrive/Desktop/ird/ClassifHackathon/feature_matrix.csv'  
    features.to_csv(output_path, index=False)
    
    print(f"Features extracted and saved to {output_path}")
    print(f"Shape of feature matrix: {features.shape}")
    print("\nFirst few rows of the feature matrix:")
    print(features.head())
