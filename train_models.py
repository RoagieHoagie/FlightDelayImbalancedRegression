import lightgbm as lgb
import numpy as np
import pickle
import os
import shutil
from datetime import datetime
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.base import clone
from tqdm import tqdm
from evaluate import evaluate, load_data, THETA

TRAINED_DIR = 'trained_cv'
EVALUATED_DIR = 'evaluated_cv'

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


def _restore_dtypes(df, reference_dtypes):
    """
    Cast df columns back to the dtypes recorded in reference_dtypes.
    iloc / concat can silently widen numeric columns to object; this undoes that.
    """
    return df.astype(reference_dtypes).infer_objects()


def train_one_model(model_name, model, strategy, Xr, yr, phi_r):
    Xr = Xr.infer_objects()
    supports_weights = model_name in ('RandomForest', 'LightGBM')
    if strategy == 'WERCS' and supports_weights:
        model.fit(Xr, yr, sample_weight=phi_r)
    else:
        model.fit(Xr, yr)
    return model


def run_model_across_strategies(
    X_test, y_test, datasets, phi_test,
    model_name, sample=1.0, strategies=[],
    n_splits=5, n_repeats=2,
):
    t0  = datetime.now()
    T_R = 0.80
    sample_tag = str(sample).replace(".", "_")
    partial_dir = f'{TRAINED_DIR}/partial_{model_name}_sample{sample_tag}'
    os.makedirs(partial_dir, exist_ok=True)
    os.makedirs(TRAINED_DIR,   exist_ok=True)
    os.makedirs(EVALUATED_DIR, exist_ok=True)

    filtered = {k: v for k, v in datasets.items() if not strategies or k in strategies}

    done_strategies = {
        f.replace('_metrics.pkl', '').replace('_model.pkl', '')
        for f in os.listdir(partial_dir)
        if f.endswith('_metrics.pkl')
    }

    bar = tqdm(filtered.items(), desc=f'{model_name}', leave=True)

    for strategy, (Xr, yr) in bar:
        partial_model_path   = os.path.join(partial_dir, f'{strategy}_model.pkl')
        partial_metrics_path = os.path.join(partial_dir, f'{strategy}_metrics.pkl')

        if strategy in done_strategies:
            bar.set_postfix_str(f'{strategy} [skipped]')
            continue

        if sample < 1.0:
            is_rare = (yr.rank(pct=True) >= T_R).astype(int)
            Xr_fit, _, yr_fit, _ = train_test_split(
                Xr, yr,
                train_size=max(sample, 0.0),
                stratify=is_rare,
                random_state=42,
            )
            Xr_fit = Xr_fit.reset_index(drop=True).astype(Xr.dtypes.to_dict())
            yr_fit = yr_fit.reset_index(drop=True)
        else:
            Xr_fit = Xr.copy()
            yr_fit = yr.copy()

        # Snapshot dtypes AFTER the subsample so CV splits can restore them
        original_dtypes = Xr_fit.dtypes.to_dict()

        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        cv_metrics_list = []

        for train_idx, val_idx in rkf.split(Xr_fit):
            X_train_cv = _restore_dtypes(Xr_fit.iloc[train_idx].reset_index(drop=True), original_dtypes)
            X_val_cv   = _restore_dtypes(Xr_fit.iloc[val_idx].reset_index(drop=True),   original_dtypes)
            y_train_cv = yr_fit.iloc[train_idx].reset_index(drop=True)
            y_val_cv   = yr_fit.iloc[val_idx].reset_index(drop=True)

            phi_train_cv = y_train_cv.rank(pct=True)
            phi_val_cv   = y_val_cv.rank(pct=True)

            cv_model = clone(models[model_name])
            cv_model = train_one_model(model_name, cv_model, strategy,
                                       X_train_cv, y_train_cv, phi_train_cv)

            cv_metrics_list.append(
                evaluate(cv_model, X_val_cv, y_val_cv, phi_val_cv, theta=THETA)
            )

        avg_cv_metrics = {k: np.mean([m[k] for m in cv_metrics_list]) for k in cv_metrics_list[0].keys()}
        print(
            f'\n[{model_name}][{strategy}] CV avg  '
            f"MAE={avg_cv_metrics['MAE']:.2f}  "
            f"F1φ={avg_cv_metrics['f1_phi']:.4f}  "
            f"precφ={avg_cv_metrics['prec_phi']:.4f}  recφ={avg_cv_metrics['rec_phi']:.4f}"
        )

        phi_fit = yr_fit.rank(pct=True)
        model = clone(models[model_name])
        model = train_one_model(model_name, model, strategy, Xr_fit, yr_fit, phi_fit)

        test_metrics = evaluate(model, X_test, y_test, phi_test, theta=THETA)
        print(
            f'[{model_name}][{strategy}] TEST    '
            f"MAE={test_metrics['MAE']:.2f}  "
            f"F1φ={test_metrics['f1_phi']:.4f}  "
            f"precφ={test_metrics['prec_phi']:.4f}  recφ={test_metrics['rec_phi']:.4f}"
        )

        with open(partial_model_path,   'wb') as f: pickle.dump(model,        f)
        with open(partial_metrics_path, 'wb') as f: pickle.dump(test_metrics, f)

    trained_model, model_results = {}, {}
    for strategy in (strategies if strategies else datasets):
        with open(os.path.join(partial_dir, f'{strategy}_model.pkl'),   'rb') as f:
            trained_model[strategy] = pickle.load(f)
        with open(os.path.join(partial_dir, f'{strategy}_metrics.pkl'), 'rb') as f:
            model_results[strategy] = pickle.load(f)

    with open(f'{TRAINED_DIR}/{model_name}_sample{sample_tag}.pkl',   'wb') as f:
        pickle.dump(trained_model, f)
    with open(f'{EVALUATED_DIR}/{model_name}_sample{sample_tag}.pkl', 'wb') as f:
        pickle.dump(model_results, f)

    shutil.rmtree(partial_dir)

    elapsed = datetime.now() - t0
    print(f'  {model_name} done in {elapsed}\n')
    print(f'  Saved {model_name}')


if __name__ == "__main__":
    os.makedirs(EVALUATED_DIR, exist_ok=True)
    os.makedirs(TRAINED_DIR,   exist_ok=True)

    datasets, X_test, y_test, phi_test = load_data()

    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'LinearRegression')
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'LinearRegression', 0.1)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'LinearRegression', 0.25)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'LinearRegression', 0.5)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'SVR', 0.00005)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'SVR', 0.001, ['SMOTER'])
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'SVR', 0.00001)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'SVR', 0.0001)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'SVR', 0.0002)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'SVR2', 0.1)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'SVR2', 0.05)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'SVR2', 0.3)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'SVR2', 0.2)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'RandomForest', 0.05)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'RandomForest', 0.01)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'RandomForest', 0.2)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'RandomForest', 0.1)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'LightGBM', 0.25)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'LightGBM', 0.5)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'LightGBM', 0.1)
    # run_model_across_strategies(X_test, y_test, datasets, phi_test, 'LightGBM', 1.0)
