from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from train import train_model

def evaluate():
    model, X_test, y_test = train_model()
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)

if __name__ == "__main__":
    evaluate()
