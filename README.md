# KARE-GAT

A notebook-based workflow for predicting **rare-earth extraction efficiency** from molecular graphs and experimental conditions, combining a **Graph Attention Network (GAT)** with a **Kolmogorov–Arnold Network (KAN)**.

The repository covers molecular graph construction, Bayesian hyperparameter optimization, leave-one-molecule-out (LOMO) cross-validation, an output-dimension ablation study, and ensemble prediction.

## Model overview

1. **Molecular representation:** atoms are graph nodes, with atomic number and RESP charge as node features. Molecular bonds define the graph edges.
2. **Graph encoder:** multi-head GAT layers, batch normalization, ELU activations, and dropout produce atom embeddings, followed by global mean pooling.
3. **Condition-aware regression:** the pooled graph representation is concatenated with seven experimental descriptors and passed to a `MultKAN` regressor to predict extraction efficiency, E (%).
4. **Ensemble prediction:** predictions from LOMO fold models are transformed back to the original target scale and averaged for each sample.

## Repository structure

```text
.
├── 01_preprocessing/
│   └── build_pyg_dataset.ipynb
├── 02_datasets/                         # Generated PyTorch Geometric datasets
├── 03_optimization/
│   ├── bayesian_optimization.ipynb
│   └── optimization_history.csv        # Included historical trial log
├── 04_cross_validation/
│   ├── lomo_cross_validation.ipynb
│   └── ablation_gat_out3.ipynb
├── 05_prediction/
│   └── lomo_ensemble_prediction.ipynb
├── environment.yml
└── README.md
```

**Included:** five notebooks, a historical optimization log containing 3,000 trials, and an exported Conda environment. Raw molecular files, experimental spreadsheets, processed datasets, and trained checkpoints must be supplied or generated separately.

## Environment setup

The supplied [`environment.yml`](environment.yml) records a **Linux / Python 3.10 / CUDA 12.1** environment. Key packages include PyTorch 2.2.2, PyTorch Geometric 2.8.0, pykan 0.2.8, RDKit, Optuna, pandas, and scikit-learn.

From the repository root on Linux, create and activate the environment:

```bash
PIP_FIND_LINKS=https://data.pyg.org/whl/torch-2.2.2+cu121.html conda env create -f environment.yml
conda activate pykan028
python -m pip install jupyterlab
python -m ipykernel install --user --name pykan028 --display-name "KARE-GAT"
jupyter lab
```

The extra wheel index provides the CUDA-specific PyTorch Geometric extensions listed in the environment file. The export contains Linux-specific build pins.

Select the **KARE-GAT** kernel and execute each notebook from top to bottom. Paths are resolved relative to the repository; the kernel working directory should be either the repository root or the notebook's own directory.

**Device configuration:** the cross-validation, ablation, and prediction notebooks currently select `cuda:7`. Change this to an available GPU, such as `cuda:0`. For CPU execution, also skip or guard the `torch.cuda.get_device_name(0)` diagnostic in the optimization and training notebooks.

## Data preparation

Place the input files under `01_preprocessing/`:

```text
01_preprocessing/
├── build_pyg_dataset.ipynb
├── experiments.xlsx
└── mol2/
    └── <molecule_name>/
        ├── <molecule_name>-<conformer_id>.mol2
        └── <molecule_name>-<conformer_id>.chg
```

- Each `.mol2` file requires a matching `.chg` file with the same stem. The charge file uses element symbols in its first column and RESP charges in its last column; non-hydrogen atom order must match the molecular structure.
- The first worksheet of `experiments.xlsx` must contain molecule names in **column A**, experimental descriptors in **columns B–M**, and extraction efficiency in **column N**.
- The preprocessing notebook excludes the rare-earth name column, the organic-phase solvent/modifier MPI columns, and their volume-ratio columns using exact spreadsheet headers in `excluded_condition_cols`. Adjust that list if your headers differ.
- The retained numeric descriptors must match the model's seven-condition input, in the order used by `CONDITION_COLUMN_NAMES` in the prediction notebook: saponification, extractant concentration, solvent MPI, rare-earth atomic number, RE³⁺ radius, RE³⁺ electron affinity, and rare-earth concentration.
- Molecule directory names must match the spreadsheet names. Use `MOLECULE_DIR_NAME_ALIASES` for explicit name mappings.

Running [`build_pyg_dataset.ipynb`](01_preprocessing/build_pyg_dataset.ipynb) pairs every experimental row with all available conformers of its molecule and writes:

```text
02_datasets/kare_gat_dataset.pt
```

The saved object is a list of PyTorch Geometric `Data` objects:

| Field | Description |
| --- | --- |
| `x`, `atom_types_chg` | Node features with shape `[num_atoms, 2]` |
| `edge_index` | Directed adjacency indices with shape `[2, num_edges]` |
| `experiment_condition_tensor` | Numeric experimental descriptors, expected shape `[1, 7]` |
| `extraction_rate` | Extraction efficiency, shape `[1, 1]` |
| `name`, `conformer_id` | Molecule and conformer identifiers |
| `experiment_condition` | Experimental descriptors as a dictionary |
| `mol2_path`, `chg_path`, `excel_row_index` | Source-file and spreadsheet-row metadata |

## Workflow

### 1. Bayesian hyperparameter optimization (optional)

Run [`bayesian_optimization.ipynb`](03_optimization/bayesian_optimization.ipynb) to search GAT layer sizes, attention heads, graph output dimensions, KAN architecture and spline parameters, and the learning rate using Optuna's TPE sampler. Defaults are **3,000 trials**, **300 epochs per trial**.

New trial results are saved to `03_optimization/optimization_results.csv`. The included `optimization_history.csv` is a historical record. Its objective values come from sample-level splitting, whereas the evaluation below holds out entire molecules.

### 2. Leave-one-molecule-out cross-validation

Run [`lomo_cross_validation.ipynb`](04_cross_validation/lomo_cross_validation.ipynb). Each fold holds out one molecule, including all its conformers and experimental rows. Node features, conditions, and targets are standardized using statistics fitted only on that fold's training data.

Before running, review `EXCLUDED_MOLECULE_NAMES`: Adapt the list to your dataset, or set it to `[]` to include all molecules.

The baseline uses GAT hidden dimensions `[64, 96, 16]`, four attention heads, and a five-dimensional graph output. Training consists of **500 epochs** with AdamW at approximately `0.004473`, followed by **50 epochs** at `0.0001`, starting from the selected first-stage checkpoint. Hyperparameters are configured directly in the notebook; rerunning optimization does not automatically update them.

Outputs are saved under `04_cross_validation/lomo_results/fold_<index>_<molecule>/`, including train/validation datasets, best checkpoints from both stages, Excel loss tables, and SVG loss and prediction plots. Checkpoints contain model weights, architecture settings, and the fold-specific standardization statistics.

### 3. Output-dimension ablation (optional)

Run [`ablation_gat_out3.ipynb`](04_cross_validation/ablation_gat_out3.ipynb) to reduce the GAT graph representation from **five dimensions to three** while retaining the baseline training procedure and other hyperparameters. Results are written to `04_cross_validation/ablation_gat_out3_results/`.

### 4. Ensemble prediction

Run [`lomo_ensemble_prediction.ipynb`](05_prediction/lomo_ensemble_prediction.ipynb) after generating the baseline LOMO checkpoints.

- `MODEL_BASE_DIR` defaults to `04_cross_validation/lomo_results/`.
- `DATASET_PATH` defaults to `02_datasets/kare_gat_dataset.pt`; change it to use another dataset with the same schema.
- Set `MOLECULE_NAMES` to select molecules, or leave it empty to predict all molecules in the dataset.
- Each fold directory must contain exactly one second-stage checkpoint matching `best_model_fold<index>-500-550-best_epoch*.pt`. Update this pattern if you change the training-stage lengths.

The notebook uses every discovered fold model, applies its saved scalers, and averages predictions on the original efficiency scale. It also exports graph-level representations and per-molecule summaries to:

```text
05_prediction/ensemble_results/LOMO_ensemble_all_prediction_results.xlsx
```

The workbook contains `Predictions` and `Summary` sheets with sample-level predictions, residuals, graph features, and molecule-level R² values. The current prediction notebook expects reference `extraction_rate` values for these reports. When predicting molecules used in cross-validation, ensemble scores include models trained on those molecules and should be interpreted separately from held-out LOMO validation scores.
