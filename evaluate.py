
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import pickle
import os


THETA = 30

samples = {
    'LinearRegression': 1.0,
    'SVR':              0.0001,
    'SVR2':             0.2,
    'LightGBM':         0.5,
    'RandomForest':     0.1,
}


def load_data():
    print("Loading data from disk...")
    pipeline_inputs = joblib.load('pipeline_inputs.joblib')

    datasets = pipeline_inputs['datasets']
    X_test   = pipeline_inputs['X_test']
    y_test   = pipeline_inputs['y_test']
    phi_test = pipeline_inputs['phi_test']

    print("Successfully loaded data!")
    return datasets, X_test, y_test, phi_test


def utility_metrics(y_true, y_pred, phi_true, theta=30.0):
    """
    Utility-based precision, recall and F1 for imbalanced regression.

    Parameters
    ----------
    y_true  : true target values
    y_pred  : predicted values
    phi_true: relevance scores for y_true (rank-based, range 0-1)
    theta   : acceptable error tolerance in minutes (default 30)
    """
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    phi     = np.array(phi_true)

    # phi for predictions (rank among predicted values)
    phi_pred = pd.Series(y_pred).rank(pct=True).values

    within_tolerance = (np.abs(y_true - y_pred) <= theta).astype(float)

    prec_denom = phi_pred.sum()
    rec_denom  = phi.sum()

    prec = (phi_pred * within_tolerance).sum() / prec_denom if prec_denom > 0 else 0.0
    rec  = (phi      * within_tolerance).sum() / rec_denom  if rec_denom  > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return dict(prec_phi=prec, rec_phi=rec, f1_phi=f1)


def evaluate(model, X_t, y_t, phi_t, theta=30.0):
    preds = np.clip(model.predict(X_t), 0, None)

    rmse = np.sqrt(mean_squared_error(y_t, preds))
    mae  = mean_absolute_error(y_t, preds)

    mask = y_t > 60
    tail_mae  = mean_absolute_error(y_t[mask], preds[mask]) if mask.sum() > 0 else np.nan
    tail_rmse = np.sqrt(mean_squared_error(y_t[mask], preds[mask])) if mask.sum() > 0 else np.nan

    weighted_mae = (phi_t * (y_t - preds).abs()).sum() / phi_t.sum()

    util = utility_metrics(y_t, preds, phi_t, theta=theta)

    return dict(
        RMSE=rmse, MAE=mae,
        Tail_RMSE=tail_rmse, Tail_MAE=tail_mae, tail_n=int(mask.sum()),
        Weighted_MAE=weighted_mae,
        **util
    )


def run_evaluations(X_test, y_test, phi_test):
    for model_name, sample in samples.items():
        sample_str = str(sample).replace('.', '_')
        filename   = f'{model_name}_sample{sample_str}.pkl'

        with open(filename, 'rb') as f:
            saved = pickle.load(f)

        model_results = {}
        for strategy, model in saved.items():
            metrics = evaluate(model, X_test, y_test, phi_test, theta=THETA)
            model_results[strategy] = metrics

        with open(f'evaluation/{filename}', 'wb') as f:
            pickle.dump(model_results, f)

        print(f'Evaluated: {filename}')


if __name__ == "__main__":
    os.makedirs('evaluation', exist_ok=True)
    datasets, X_test, y_test, phi_test = load_data()
    run_evaluations(X_test, y_test, phi_test)
