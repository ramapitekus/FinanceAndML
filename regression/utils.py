from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

def perform_statistics(y_pred, y_test):
    print("Statistics")

    mrse = np.sqrt(mean_squared_error(y_pred, y_test))
    print(f"rmse is {mrse}")

    mae = mean_absolute_error(y_test,y_pred)
    print(f"mae is {mae}")

    mape = mean_absolute_percentage_error(y_test,y_pred)
    print(f"mape is {mape}")

    r2=r2_score(y_test,y_pred)
    print(f"R2 is {r2}\n")
