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
- Notes: This function loads data from a file into a pandas DataFrame, optionally dropping specified columns. The function now also prints the number of records read from the item path.

### predict.py

#### get_args()

- Parameters: None
- Returns: An argparse.Namespace object that contains the command-line arguments.
- Errors/Exceptions: Raises an argparse.ArgumentError if the command-line arguments are not valid.
- Example: `args = get_args()`
- Notes: This function parses command-line arguments. It now uses double quotes instead of single quotes for string literals.

#### xy(data_set)

- Parameters:
  - data_set (DataFrame): The data set.
- Returns: A tuple containing the features (X) and the target (Y).
- Errors/Exceptions: None
- Example: `X, Y = xy(data_set)`
- Notes: This function separates the features and the target from the data set. It now checks for the "index" column in `X` and drops it if it exists.

#### predict(model, data, X, Y)

- Parameters:
  - model (str): The path to the trained model.
  - data (DataFrame): The data set.
  - X (DataFrame): The features.
  - Y (Series): The target.
- Returns: A list containing the predictions.
- Errors/Exceptions: Raises a FileNotFoundError if the model does not exist.
- Example: `predictions = predict(model, data, X, Y)`
- Notes: This function makes predictions using a trained model. It now prints the accuracy, recall, and precision scores.

#### is_func_header(line)

- Parameters:
  - line (str): The line to check.
- Returns: A boolean indicating whether the line is a function header.
- Errors/Exceptions: None
- Example: `is_header = is_func_header(line)`
- Notes: This function checks whether a line is a function header. It now uses double quotes instead of single quotes for string literals.

#### is_func_body(line)

- Parameters:
  - line (str): The line to check.
- Returns: A boolean indicating whether the line is a function body.
- Errors/Exceptions: None
- Example: `is_body = is_func_body(line)`
- Notes: This function checks whether a line is a function body. It now uses double quotes instead of single quotes for string literals.

#### check_line(line, errors)

- Parameters:
  - line (str): The line to check.
  - errors (dict): A dictionary containing the errors.
- Returns: The line with any errors marked.
- Errors/Exceptions: None
- Example: `line = check_line(line, errors)`
- Notes: This function checks a line for errors and marks any errors found. It now uses double quotes instead of single quotes for string literals.

#### mark_errors(asm_source, errors, asm_out)

- Parameters:
  - asm_source (str): The path to the assembly source file.
  - errors (dict): A dictionary containing the errors.
  - asm_out (str): The path to the output file.
- Returns: None
- Errors/Exceptions: Raises a FileNotFoundError if the assembly source file does not exist.
- Example: `mark_errors(asm_source, errors, asm_out)`
- Notes: This function marks errors in the assembly source file. It now uses double quotes instead of single quotes for string literals.

#### predict_summary(sum_prediction, data, threshold)

- Parameters:
  - sum_prediction (list): The list of predictions.
  - data (DataFrame): The data set.
  - threshold (int): The threshold for making a prediction.
- Returns: A dictionary containing the errors.
- Errors/Exceptions: None
- Example: `errors = predict_summary(sum_prediction, data, threshold)`
- Notes: This function summarizes the predictions and returns any errors. It now uses double quotes instead of single quotes for string literals.

#### main()

- Parameters: None
- Returns: None
- Errors/Exceptions: Raises an argparse.ArgumentError if the command-line arguments are not valid.
- Example: `main()`
- Notes: This function uses the command line arguments to predict the target values using the provided models, summarizes the predictions, and marks the errors in the assembly source file.

### compare_crt_features.py

#### main()

- Parameters: None
- Returns: None
- Errors/Exceptions: Raises an argparse.ArgumentError if the command-line arguments are not valid.
- Example: `main()`
- Notes: This function parses command-line arguments, loads data, compares CRT features, and outputs patterns. It now uses double quotes instead of single quotes for string literals.

### inst_analysis.py

#### load_data(data_folder, trial_mode)

- Parameters:
  - data_folder (str): The path to the data folder.
  - trial_mode (bool): Whether to run in trial mode.
- Returns: A pandas DataFrame.
- Errors/Exceptions: Raises a FileNotFoundError if the data_folder does not exist.
- Example: `df = load_data('path/to/data', True)`
- Notes: This function loads data from a folder into a pandas DataFrame, optionally running in trial mode. It now uses double quotes instead of single quotes for string literals.

#### train(data_set, mdepth, mleaf)

- Parameters:
  - data_set (DataFrame): The data set.
  - mdepth (int): The maximum depth of the decision tree.
  - mleaf (int): The minimum number of samples required to be at a leaf node.
- Returns: The best model.
- Errors/Exceptions: None
- Example: `best_model = train(data_set, mdepth, mleaf)`
- Notes: This function trains a decision tree classifier on the data set. It now uses double quotes instead of single quotes for string literals.

#### main()

- Parameters: None
- Returns: None
- Errors/Exceptions: Raises an argparse.ArgumentError if the command-line arguments are not valid.
- Example: `main()`
- Notes: This function parses command-line arguments, loads data, trains a decision tree classifier, tests the classifier, and outputs the results. It now uses double quotes instead of single quotes for string literals.
