import pandas as pd
import requests
from io import StringIO
from tqdm import tqdm
import os

# Define the project root directory
PROJECT_ROOT = '/mnt/c/Users/Linda/Desktop/GitHub_Projects/MLP_ProteinLocalization'

# Define paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw_data.tab')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed_data.csv')

def get_sequences_from_uniprot(uniprot_ids):
    base_url = "https://rest.uniprot.org/uniprotkb/stream"
    batch_size = 100
    sequences = {}
    
    for i in tqdm(range(0, len(uniprot_ids), batch_size), desc="Fetching sequences"):
        batch = uniprot_ids[i:i+batch_size]
        params = {
            "query": " OR ".join(f"accession:{acc}" for acc in batch),
            "format": "fasta"
        }
        
        response = requests.get(base_url, params=params)
        if response.ok:
            fasta_data = StringIO(response.text)
            current_id = None
            current_sequence = []
            
            for line in fasta_data:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        sequences[current_id] = ''.join(current_sequence)
                    current_id = line.split('|')[1]
                    current_sequence = []
                else:
                    current_sequence.append(line)
            
            if current_id:
                sequences[current_id] = ''.join(current_sequence)
        else:
            print(f"Failed to retrieve sequences for batch starting with {batch[0]}")
    
    return sequences

def main():
    # Load your data
    df = pd.read_csv(RAW_DATA_PATH, sep='\t', header=None, names=['UniProt_AC', 'Type', 'Length'])

    # Get unique UniProt ACs
    uniprot_ids = df['UniProt_AC'].unique().tolist()

    # Retrieve sequences
    sequences = get_sequences_from_uniprot(uniprot_ids)

    # Add sequences to the DataFrame
    df['Sequence'] = df['UniProt_AC'].map(sequences)

    # Save the updated DataFrame
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Data with sequences saved to '{PROCESSED_DATA_PATH}'")

if __name__ == "__main__":
    main()