import math
from sklearn.metrics import mean_squared_error

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=64):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

def predict_and_inverse(model, X, scaler):
    pred = model.predict(X)
    return scaler.inverse_transform(pred)

def calculate_rmse(y_true, y_pred_scaled, scaler):
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))
    return math.sqrt(mean_squared_error(y_true_inv, y_pred_scaled))
