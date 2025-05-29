import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(data_scaled, train_predict, test_predict, scaler, time_step, save_path=None):
    trainPredictPlot = np.empty_like(data_scaled)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[time_step:len(train_predict)+time_step, :] = train_predict

    testPredictPlot = np.empty_like(data_scaled)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(time_step*2)+1:len(data_scaled)-1, :] = test_predict

    plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(data_scaled), label='Actual')
    plt.plot(trainPredictPlot, label='Train Predict')
    plt.plot(testPredictPlot, label='Test Predict')
    plt.title('Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_future_predictions(data, future_preds, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(data)), data, label='Actual')
    plt.plot(np.arange(len(data), len(data)+len(future_preds)), future_preds, label='Future Prediction')
    plt.title('Future Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
