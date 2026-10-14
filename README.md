# Ling-Spam hands-on

Source kernel: https://www.kaggle.com/code/sharmoul/ling-spam-dataset-eda
Dataset: https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset

## Setup
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    scripts/download_data.sh          # add --force to re-download

Kaggle credentials (any one): ~/.kaggle/access_token, KAGGLE_API_TOKEN,
~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY. Nothing is stored in this repo.

## Layout
- notebooks/ling-spam-dataset-eda/  downloaded kernel (untrusted; not modified)
- data/raw/                         lingSpam.csv, enronSpamSubset.csv, completeSpamAssassin.csv

## Path mapping
The notebook reads `/kaggle/input/email-spam-dataset/lingSpam.csv`.
Locally use `data/raw/lingSpam.csv` (relative to the project root), e.g. replace the
path in the read_csv cell, or symlink: `sudo ln -s "$PWD/data/raw" /kaggle/input/email-spam-dataset`.
Run Jupyter from the project root: `.venv/bin/jupyter lab`.
# ling-spam-hands-on
