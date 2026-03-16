# Pathology-Prior Driven Substructure-Aware Graph Neural Network for Whole Slide Image Classification 

# Full Paper is Under Review by Scientifc Reports Now!

This repository contains code for processing Whole Slide Images (WSI) of breast cancer sentinel lymph nodes, extracting patch-level features, constructing pathology-prior driven graphs, and training a substructure-aware Graph Isomorphism Network (GIN) for classification.

## Features

- **WSI Processing**: Extract tissue patches from SVS files using OpenSlide, with automatic tissue segmentation.
- **Feature Extraction**: Use pre-trained ResNet50 to extract 2048-dimensional features from patches.
- **Graph Construction**: Build graphs from patch features with spatial relationships, enhanced with pathology-prior substructure features (triangle density, clustering coefficient, 4-cycle density, degree centrality).
- **Substructure-Aware GNN**: Modified GIN with attention mechanisms on substructure features for improved performance on pathology tasks.
- **Training and Evaluation**: Includes training scripts with cross-validation, ROC-AUC, and other metrics.

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- OpenSlide
- PyTorch Geometric
- Other dependencies: numpy, scipy, scikit-image, tqdm, networkx, sklearn, pandas

### Install Dependencies

```bash
pip install -r requirements.txt
```

For OpenSlide, you may need to install system libraries:

- Ubuntu: `sudo apt-get install openslide-tools python3-openslide`
- macOS: `brew install openslide`
- Windows: Download from [OpenSlide website](https://openslide.org/download/)

## Usage

### 1. Process WSI Files

Place your SVS files in a directory, then edit `WSI_to_npz_features.py` to set:

```python
WSI_DIR = 'path/to/your/WSI/directory'
OUTPUT_DIR = 'path/to/your/output/directory'
```

Then run:

```bash
python WSI_to_npz_features.py
```

This will extract features and save them as NPZ files in the specified output directory.

### 2. Build Graphs

Edit `features_to_graphs.py` to set:

```python
feature_path = 'path/to/your/feature/files'
output_path = 'path/to/your/output/directory'
```

Then run:

```bash
python features_to_graphs.py
```

This constructs graphs with substructure features and saves them as PKL files in the specified output directory.

### 3. Train the Model

Edit `train.py` to set:

```python
bdir = 'path/to/your/graph/files'
save_dir = 'path/to/your/saved/models'
```

Then run:

```bash
python train.py
```

Adjust other parameters in the script as needed for your dataset.

## Project Structure

- `WSI_to_npz_features.py`: WSI processing and feature extraction.
- `features_to_graphs.py`: Graph construction with substructure features.
- `GNN.py`: Substructure-aware GNN model.
- `train.py`: Training script.
- `platt.py`: Platt scaling for calibration.
- `utils.py`: Utility functions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

