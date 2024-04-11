## What is EsmTemp?

EsmTemp is a repository leveraging the ESM-2 protein model for fine-tuning, aimed at improving results in melting temperature prediction. The repository utilizes the ESM-2 model, accessible [here](https://github.com/facebookresearch/esm), as a basis for fine-tuning processes. The objective is to enhance the accuracy of predictions related to melting temperatures in protein structures.

In transfer learning with protein language models, the embeddings extracted from these latent spaces serve as powerful features for downstream tasks. By leveraging the pre-trained knowledge encoded within the embeddings, this approach allows us to fine-tune the model for predicting the melting temperatures of proteins.

![Alt text](figures/conception.png)

## Data

The dataset was compiled by integrating data from two sources of protein thermostability information: ProThermDB and Meltome Atlas.
Tm of proteins was achieved by only keeping measurements conducted on cell lysates. The dataset was refined by removing protein sequences exhibiting Tm values outside the range of 30°C to 98°C, as well as sequences with lengths below 100 or exceeding 900 amino acids, to maintain a focus on proteins of relevant stability and size for accurate modeling. Duplicate entries were identified and eliminated, and the remaining sequences were clustered to further reduce redundancy with the CD-HIT algorithm, applying a sequence similarity threshold of 0.7 and a minimum cluster size of 3, while all other parameters were left at their default settings. This process resulted in a curated dataset consisting of 24,472 unique amino acid sequences, each annotated with its corresponding source organism and Tm value.

The distributions of temperature and protein sequence lengths within the dataset are illustrated below.

![Alt text](figures/distribution.png)

In the paper, we also emphasize fine-tuning the model exclusively for proteins from specific organisms, particularly the most frequently occurring ones. The figure below visually illustrates the distribution of melting temperatures for individual proteins originating from the four most common organisms in the curated training set. Histogram depicting the distribution of melting temperatures (Tm) across a protein sequence dataset from various organisms’ lysates: Human (A), Arabidopsis thaliana seedling (B), Caenorhabditis elegans (C), Mus musculus BMDC (D)

![Alt text](figures/tm_distribution.png)

The data are provided in this repo in file dataset.csv

## Results

Our results highlight that the model captures the general trend of Tm for a vast landscape of protein sequence, achieving a coefficient of determination R<sup>2</sup> of 0.70 and a mean absolute error of 4.3°C.

Figures below shows the correlation analysis depicts the relationship between melting temperatures with the prediction for the best fine-tune ESM-2 model across specific organisms.

![Alt text](figures/results_all.png)

## How to create conda environment?

```bash
conda create --name esm_env python=3.9
conda activate esm_env
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pandas
pip install wandb
pip install fair-esm
pip install pyarrow
```

or alternatively, use an environment.yml file:

```bash
conda env create -f environment.yml
conda activate esm_env
```

## How to speed up training by using esm from github?

Opting to download the model directly from the [GitHub repository](https://github.com/facebookresearch/esm) can expedite the fine-tuning process, reducing the time required for model fine-tune.

## How input file should looks like?

The input CSV file is expected to have two columns: 'protein_sequence' containing strings representing amino acid sequences, and 'tm' containing float values indicating temperatures.

## How to run EsmTemp?

To illustrate the usage of EsmTemp, consider the following example command:

```bash
python run_training.py --train_csv_file "data/example_files/train_1.csv" --val_csv_file "data/example_files/val_1.csv" 
```

If you wish to incorporate WandB for experiment tracking, you can add the following option to the command:

```bash
python run_training.py --train_csv_file "data/example_files/train_1.csv" --val_csv_file "data/example_files/val_1.csv" --use_wandb
```

## Generate row ESM2 embedding

To generate ESM2 embeddings from the 650M model, utilize the generate_row_embedding.py script as follows:

```bash
python generate_row_embedding.py --model_size "big" --dataset_path "data/example_files/val_1.csv" --path_embedding_file "val_1_embeddings.csv"
```

Ensure that the CSV input file follows the same format as mentioned above. The path_embedding_file should have a CSV file extension. Subsequently, use this generated file as an input file for training.

## CD-hit filtering for similar sequences

In case you want to use cd-hit, you need to install Biopython:

```bash
conda install -c conda-forge biopython
```

To illustrate the usage of cd-hit, consider the following example command:

```bash
python run_cd_hit.py --input_csv_file "data/example_files/train_1.csv" 
```

Locate the filtered file at: data/data_files/{file_name}_07_n3.csv.

By default, the script employs a sequence similarity threshold of 0.7 and sets the neighboring 'n' value to 3. To adjust these parameters, use the following command:

```bash
python run_cd_hit.py --input_csv_file "data/example_files/train_1.csv" --treshold 0.8 --n 2
```
