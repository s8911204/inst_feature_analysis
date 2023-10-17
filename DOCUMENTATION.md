# inst_feature_analysis Repository

## Overview

This repository contains Python scripts for analyzing and predicting data. It includes scripts for loading and preprocessing data, training machine learning models, making predictions, and comparing features.

## Installation

1. Ensure you have Python 3 installed.
2. Clone the repository: `git clone https://github.com/user/inst_feature_analysis.git`
3. Install the required libraries: `pip install pandas matplotlib sklearn argparse joblib os json numpy shutil sys collections pydotplus`

## File Structure

- `analyze_debug.py`: This script analyzes debug data.
- `compare_crt_features.py`: This script compares CRT features.
- `dot_graph.py`: This script generates dot graphs.
- `inst_analysis.py`: This script analyzes instruction data.
- `predict.py`: This script makes predictions using trained models.
- `utils.py`: This script contains utility functions for data loading and preprocessing.

## Usage

- `python analyze_debug.py [arguments]`
- `python compare_crt_features.py [arguments]`
- `python dot_graph.py [arguments]`
- `python inst_analysis.py [arguments]`
- `python predict.py [arguments]`

## API References

### utils.py

#### toDf(item_path, dropCols=None)

- Parameters:
  - item_path (str): The path to the item.
  - dropCols (list, optional): The columns to drop.
- Returns: A pandas DataFrame.
- Errors/Exceptions: Raises a FileNotFoundError if the item_path does not exist.
- Example: `df = toDf('path/to/item', ['column1', 'column2'])`
- Notes: This function loads data from a file into a pandas DataFrame, optionally dropping specified columns.

### predict.py

#### main()

- Parameters: None
- Returns: None
- Errors/Exceptions: Raises an argparse.ArgumentError if the command-line arguments are not valid.
- Example: `main()`
- Notes: This function parses command-line arguments, loads data, makes predictions using multiple models, and marks errors.

### compare_crt_features.py

#### main()

- Parameters: None
- Returns: None
- Errors/Exceptions: Raises an argparse.ArgumentError if the command-line arguments are not valid.
- Example: `main()`
- Notes: This function parses command-line arguments, loads data, compares CRT features, and outputs patterns.

### inst_analysis.py

#### load_data(data_folder, trial_mode)

- Parameters:
  - data_folder (str): The path to the data folder.
  - trial_mode (bool): Whether to run in trial mode.
- Returns: A pandas DataFrame.
- Errors/Exceptions: Raises a FileNotFoundError if the data_folder does not exist.
- Example: `df = load_data('path/to/data', True)`
- Notes: This function loads data from a folder into a pandas DataFrame, optionally running in trial mode.
