import lightgbm as lgb
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm
from .evaluate import evaluate, load_data, THETA


models = {
    'LinearRegression': LinearRegression(n_jobs=-1),
    'SVR': Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=5.0, epsilon=10.0, cache_size=10000))
    ]),
    'SVR2': Pipeline([
        ('scaler',    StandardScaler()),
        ('nystroem',  Nystroem(kernel='rbf', gamma=0.1, n_components=300, random_state=42)),
        ('sgd',       SGDRegressor(loss='epsilon_insensitive', epsilon=10.0,
                                alpha=0.001, max_iter=1000, random_state=42))
    ]),
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=42
    ),
    'LightGBM': lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
}


def train_one_model(model_name, model, strategy, Xr, yr, phi_r):
    Xr = Xr.infer_objects()  # restore dtypes if corrupted by concat/iloc
    supports_weights = model_name in ('RandomForest', 'LightGBM')
    if strategy == 'WERCS' and supports_weights:
        model.fit(Xr, yr, sample_weight=phi_r)
    else:
        model.fit(Xr, yr)
    return model


def run_model_across_strategies(X_test, y_test, datasets, phi_test, model_name, sample=1.0):
    t0  = datetime.now()
    T_R = 0.80

    bar = tqdm(datasets.items(), desc=f'{model_name}', leave=True)
    print(f"Started training {model_name}")

    model_results = {}
    trained_model = {}

    for strategy, (Xr, yr) in bar:
        phi_r      = yr.rank(pct=True)
        rare_keep   = phi_r[phi_r >= T_R].index.to_series().sample(frac=sample, random_state=42).values
        normal_keep = phi_r[phi_r <  T_R].index.to_series().sample(frac=sample, random_state=42).values
        keep        = np.concatenate([rare_keep, normal_keep])

        Xr_fit = Xr.iloc[keep].reset_index(drop=True).astype(Xr.dtypes.to_dict())
        yr_fit = yr.iloc[keep].reset_index(drop=True)
        phi_r  = yr_fit.rank(pct=True)

        bar.set_postfix_str(f'{strategy} ({len(Xr_fit):,} / {len(Xr):,} rows)')

        model = models[model_name]
        model = train_one_model(model_name, model, strategy, Xr_fit, yr_fit, phi_r)
        trained_model[strategy] = model

        metrics = evaluate(model, X_test, y_test, phi_test, theta=THETA)       
        model_results[strategy] = metrics

        print(
            f'  [{model_name}][{strategy}] '
            f"MAE={metrics['MAE']:.2f}  F1φ={metrics['f1_phi']:.4f}  "
            f"precφ={metrics['prec_phi']:.4f}  recφ={metrics['rec_phi']:.4f}"
        )


    elapsed = datetime.now() - t0
    print(f'  {model_name} done in {elapsed}\n')

    with open(f'evaluation/{model_name}_sample{str(sample).replace(".", "_")}.pkl', 'wb') as f:
        pickle.dump(model_results, f)
    with open(f'{model_name}_sample{str(sample).replace(".", "_")}.pkl', 'wb') as f:
        pickle.dump(trained_model, f)
    print(f'  Saved {model_name}')



if __name__ == "__main__":
    os.makedirs('evaluation', exist_ok=True)
    os.makedirs('trained', exist_ok=True)

    datasets, X_test, y_test, phi_test = load_data()

    run_model_across_strategies(X_test, y_test, datasets, phi_test, 'LinearRegression')
    run_model_across_strategies(X_test, y_test, datasets, phi_test, 'SVR', 0.0001)
    run_model_across_strategies(X_test, y_test, datasets, phi_test, 'SVR2', 0.2)
    run_model_across_strategies(X_test, y_test, datasets, phi_test, 'RandomForest', 0.1)
    run_model_across_strategies(X_test, y_test, datasets, phi_test, 'LightGBM', 0.5)
