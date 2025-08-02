import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from utils.data_utils import load_and_scale_data, create_dataset
from model.model_utils import build_transformer_model
from utils.train_utils import train_model, predict_and_inverse, calculate_rmse
from utils.plot_utils import plot_predictions, plot_future_predictions
import pandas.errors

def run_pipeline_for_all_targets(csv_path, output_dir, target_columns, time_step=100, epochs=50, batch_size=64, future_days=30):
    csv_base = os.path.splitext(os.path.basename(csv_path))[0]
    try:
        df = pd.read_csv(csv_path)
    except pandas.errors.EmptyDataError:
        print(f"Skipping empty file: {csv_path}")
        return
    
    predictions_dir = os.path.join('predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    plot_dir = os.path.join(predictions_dir, csv_base)
    
    all_future_preds = {}
    all_test_predictions = {}  # Store test predictions for each column
    for col in target_columns:
        if col not in df.columns or df[col].isnull().all():
            print(f"Skipping {col} (not found or all values are NaN)")
            continue
        if df[col].isnull().any():
            print(f"Skipping {col} (contains NaN values)")
            continue
        data = df[[col]].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        training_size = int(len(data_scaled) * 0.67)
        train_data, test_data = data_scaled[0:training_size,:], data_scaled[training_size:len(data_scaled),:]
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)
        if X_train.size == 0 or X_test.size == 0:
            print(f"Skipping {col} (not enough data after time_step={time_step})")
            continue
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        model = build_transformer_model((X_train.shape[1], X_train.shape[2]))
        train_model(model, X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
        train_predict = predict_and_inverse(model, X_train, scaler)
        test_predict = predict_and_inverse(model, X_test, scaler)
        
        # Store test predictions for this column
        all_test_predictions[col] = test_predict
        
        train_rmse = calculate_rmse(y_train, train_predict, scaler)
        test_rmse = calculate_rmse(y_test, test_predict, scaler)
        print(f"{col} - Train RMSE: {train_rmse}")
        print(f"{col} - Test RMSE: {test_rmse}")
        
        # Create plot directory only when we have successful predictions
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
            
        plot_path = os.path.join(plot_dir, f"{col}_prediction.png")
        plot_predictions(data_scaled, train_predict, test_predict, scaler, time_step, save_path=plot_path)
        # Future prediction
        last_sequence = data_scaled[-time_step:]
        future_preds = []
        current_seq = last_sequence.copy()
        for _ in range(future_days):
            pred = model.predict(current_seq.reshape(1, time_step, 1))
            future_preds.append(float(scaler.inverse_transform([[pred[0,0]]])[0,0]))
            current_seq = np.append(current_seq[1:], [[pred[0,0]]], axis=0)
        future_preds_arr = np.array(future_preds).reshape(-1,1)
        plot_future_path = os.path.join(plot_dir, f"{col}_future_{future_days}days.png")
        plot_future_predictions(data, future_preds_arr, save_path=plot_future_path)
        all_future_preds[col] = future_preds
    
    # Only save predictions if we have any successful predictions
    if all_future_preds:
        # Save all predictions for this CSV to a JSON file (future predictions)
        json_path = os.path.join(plot_dir, f"{csv_base}_future_{future_days}days.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_future_preds, f, ensure_ascii=False, indent=2)
    
    # Save test predictions if we have any
    if all_test_predictions:
        # Get the length of test predictions (should be same for all columns)
        first_col = list(all_test_predictions.keys())[0]
        test_length = len(all_test_predictions[first_col])
        
        test_results = []
        test_indices = range(len(df) - test_length, len(df))
        
        for idx, df_idx in enumerate(test_indices):
            row = df.iloc[df_idx]
            entry = {
                "date": int(row["date"]) if "date" in row else None,
                "source": int(row["source"]) if "source" in row else None,
                "ticker": int(row["ticker"]) if "ticker" in row else None,
                "actual": {},
                "prediction": {}
            }
            
            # Add actual values for all target columns
            for col in target_columns:
                if col in row and pd.notnull(row[col]):
                    entry["actual"][col] = float(row[col])
                else:
                    entry["actual"][col] = None
            
            # Add predictions for columns we actually processed
            for col in target_columns:
                if col in all_test_predictions:
                    entry["prediction"][col] = float(all_test_predictions[col][idx][0])
                else:
                    entry["prediction"][col] = None
            
            test_results.append(entry)
        
        # Add future predictions (30 days) to the test_predictions JSON
        if all_future_preds:
            # Get the last date from the dataframe
            last_date = int(df.iloc[-1]["date"]) if "date" in df.columns else None
            last_source = int(df.iloc[-1]["source"]) if "source" in df.columns else None
            last_ticker = int(df.iloc[-1]["ticker"]) if "ticker" in df.columns else None
            
            # Add future predictions (assuming daily data, increment by 86400 seconds = 1 day)
            for day in range(1, future_days + 1):
                future_entry = {
                    "date": last_date + (day * 86400) if last_date else None,
                    "source": last_source,
                    "ticker": last_ticker,
                    "actual": {},
                    "prediction": {}
                }
                
                # Set actual values to null for future predictions
                for col in target_columns:
                    future_entry["actual"][col] = None
                
                # Add future predictions for columns we actually processed
                for col in target_columns:
                    if col in all_future_preds and day - 1 < len(all_future_preds[col]):
                        future_entry["prediction"][col] = float(all_future_preds[col][day - 1])
                    else:
                        future_entry["prediction"][col] = None
                
                test_results.append(future_entry)
        
        test_json_path = os.path.join(plot_dir, f"{csv_base}.json")
        with open(test_json_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    if not all_future_preds and not all_test_predictions:
        print(f"No predictions generated for {csv_base} - all columns were skipped")
