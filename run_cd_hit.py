import subprocess
import argparse
import logging
import pandas as pd
import os
from Bio import SeqIO


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_cdhit(input_file, output_file, treshold=0.7, n="3"):
    cdhit_cmd = "cd-hit"
    cmd = [cdhit_cmd, "-i", input_file, "-o", output_file, "-c", str(treshold), "-n", n]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        logging.info("An error occurred while running CD-HIT.")
        logging.info(f"Error message: {stderr.decode()}")
    else:
        logging.info("CD-HIT execution completed successfully.")


def df_to_fasta(df, fasta_file_name):
    with open(fasta_file_name, 'w') as f:
        if 'uniprot_id' in df.columns and 'gene_name' in df.columns:
            for index, row in df.iterrows():
                uniprot_id = row['uniprot_id']
                if pd.isnull(uniprot_id):
                    gene_name = row['gene_name']
                    if pd.isnull(gene_name):
                        uniprot_id = str(index)
                    else:
                        uniprot_id = gene_name
                sequence = row['protein_sequence']
                f.write(f">{uniprot_id}\n{sequence}\n")
        else:
            for index, row in df.iterrows():
                uniprot_id = str(index)
                sequence = row['protein_sequence']
                f.write(f">{uniprot_id}\n{sequence}\n")


def fasta_to_df(fasta_file):
    uniprot_ids = []
    protein_sequences = []
    for record in SeqIO.parse(fasta_file, 'fasta'):
        uniprot_id = record.description
        uniprot_ids.append(uniprot_id)
        protein_sequences.append(str(record.seq))
    df_fasta_filtered = pd.DataFrame({'uniprot_id': uniprot_ids, 'protein_sequence': protein_sequences})
    return df_fasta_filtered


def parse_args():
    parser = argparse.ArgumentParser(description="Run cd-hit on a sequence.")

    parser.add_argument(
        "--input_csv_file",
        type=str,
        required=True,
        help="Path to the input csv file."
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold value for cd-hit. Default is 0.7."
    )

    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Value for the 'n' parameter in cd-hit. Default is 3."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    df = pd.read_csv(args.input_csv_file)
    file_name = args.input_csv_file.split("/")[-1].split(".")[0]
    df_to_fasta(df, "bioinformatics/data_files/meltome_protherm_human_fin.fasta")
    
    os.makedirs("data/data_files/", exist_ok=True)
    
    run_cdhit(input_file=f"data/data_files/{file_name}.fasta",
              output_file=f"data/data_files/{file_name}_07_n3.fasta", 
              treshold=args.treshold, n=args.n)
    
    df_after_cdhit = fasta_to_df(f"data/data_files/{file_name}_07_n3.fasta")
    
    df_after_cdhit.to_csv(f"data/data_files/{file_name}_07_n3.csv", index=False)