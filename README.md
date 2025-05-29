# Currency Price Prediction using Transformers

This project demonstrates how to predict financial time series (such as stock or currency prices) using a Transformer model implemented with TensorFlow. The pipeline is designed to work with multiple CSV files and multiple target columns, and can fetch data from an API or use local files.

## Project Overview

The project uses a simplified Transformer architecture to forecast prices based on historical data. It includes data loading, preprocessing, model training, prediction, and evaluation phases, with an emphasis on understanding the Transformer's application in time series forecasting.

## Requirements

To run this project, you need the following libraries:
- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

You can install these dependencies via pip:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

## Dataset
The dataset should be in CSV format and contain at least the following columns:

- `buyPrice`, `sellPrice`, `date`, `dateInDatetime`

You can use your own CSV files by placing them in the `content/` directory, or let the pipeline fetch data from the API. The pipeline is designed to work with any currency or financial instrument as long as these columns are present.

## Usage
1. **Data Preparation:** Place your CSV files in the `content/` directory, or let the script fetch them from the API.
2. **Model Training:** Run the main script:
   ```bash
   python main.py
   ```
   By default, the script will process all CSV files in `content/` and train a separate model for each selected target column (e.g., `buyPrice`, `sellPrice`, `last`).
3. **Evaluation and Visualization:** The script will evaluate the model's performance using RMSE and generate plots for each target. All results, including plots and predictions, will be saved under the `predictions/` directory.

## Code Structure
- `main.py`: Main pipeline entry point. Handles configuration and runs the full workflow.
- `utils/`: Utility modules for data loading, preprocessing, training, plotting, and data source management.
- `model/`: Model architecture (Transformer) definition.
- `content/`: Place your CSV files here.
- `predictions/`: All output plots and prediction JSON files are saved here, organized by CSV file.

## Output
- For each CSV and each selected target column:
  - Training and test prediction plots (`*_prediction.png`)
  - 30-day future prediction plots (`*_future_30days.png`)
  - 30-day future predictions as JSON (`*_future_30days.json`)
  - Test set predictions as JSON (`*_test_predictions.json`)
- All outputs are organized under `predictions/{csv_file_name}/`

## Customization
- You can change the target columns to train on by editing the `selected_columns` list in `main.py`.
- You can switch between API and local data by changing the `source_type` parameter in `GetRateData`.
- Model and training parameters (epochs, batch size, time step, etc.) can be adjusted in `main.py`.

## License
This project is open-source and available under the MIT license.
