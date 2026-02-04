import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


def remove_correlations(df, var_list):
    df = df.copy()
    df.drop(columns=var_list, inplace=True, errors="ignore")
    return df
    
def log_trainsform(df, var_list):
    df = df.copy()
    for col in var_list:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df

def drop_names(df, var_list):
    df = df.copy()
    no_names_df = df.drop(columns=var_list)
    return no_names_df

def encode_categorical(df, encoder=None):
    df = df.copy()
    if "type" not in df.columns:
        return df, encoder
    if encoder is None:
        try:
            encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop=None
        )
        except TypeError:
            encoder = OneHotEncoder(
            sparse=False,
            handle_unknown="ignore",
            drop=None
     )
        encoded = encoder.fit_transform(df[["type"]])
    else:
        encoded = encoder.transform(df[["type"]])
    # Create dataframe with proper column names
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(["type"]),
        index=df.index
    )
    # Drop original column and concat encoded
    df = df.drop(columns=["type"])
    df = pd.concat([df, encoded_df], axis=1)
    return df, encoder

def subset(df, pct):
    df = df.copy()
    subset = df.sample(frac=(pct/100), random_state=42)
    return subset

import pandas as pd


def encode_names(df, name_cols):
    df = df.copy()
    freq_maps = {}
    for col in name_cols:
        freq = df[col].value_counts()
        freq_maps[col] = freq
        df[col] = df[col].map(freq).fillna(0).astype("int32")
    return df, freq_maps

def train_lgbm_default(
    X,
    y,
    *,
    num_classes=4,
    test_size=0.2,
    random_state=42,
    use_class_weight=True,
    early_stopping_rounds=50,
    log_every=50,
    lgbm_params=None,
):

    if lgbm_params is None:
        lgbm_params = {}
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    base_params = dict(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=2000,      
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=-1,
        random_state=random_state,
    )
    if use_class_weight:
        base_params["class_weight"] = "balanced"
    base_params.update(lgbm_params)
    model = LGBMClassifier(**base_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",  # training signal; selection via Macro F1 below
        callbacks=[
            early_stopping(early_stopping_rounds),
            log_evaluation(log_every),
        ],
    )
    # Evaluate with Kaggle metric: Macro F1
    y_val_pred = model.predict(X_val)
    metrics = {
        "val_f1_macro": float(f1_score(y_val, y_val_pred, average="macro")),
        "val_f1_weighted": float(f1_score(y_val, y_val_pred, average="weighted")),
        "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators)),
    }
    return model, metrics, (X_val, y_val)


def predict_lgbm(model, X_test):
    return model.predict(X_test)


def predict_proba_lgbm(model, X_test):
    return model.predict_proba(X_test)

def train_lgbm_weighted(
    X,
    y,
    *,
    class_weight,
    num_classes=4,
    test_size=0.2,
    random_state=42,
    cv_folds=3,
    n_jobs=-1,
    param_grid=None,
    refit=True,
    verbose=1,
):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    base_model = LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        class_weight=class_weight,
        n_estimators=1000,          # grid can override
        learning_rate=0.05,         # grid can override
        n_jobs=n_jobs,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )
    if param_grid is None:
        param_grid = {
            "num_leaves": [31, 63],
            "min_child_samples": [20, 50, 100],
        }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        verbose=verbose,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_val_pred = best_model.predict(X_val)
    metrics = {
        "val_f1_macro": float(f1_score(y_val, y_val_pred, average="macro")),
        "val_f1_weighted": float(f1_score(y_val, y_val_pred, average="weighted")),
        "cv_best_score_f1_macro": float(grid.best_score_),
        "best_params": dict(grid.best_params_),
    }
    artifacts = {
        "X_val": X_val,
        "y_val": y_val,
        "grid": grid,
    }
    return best_model, metrics, artifacts

# Weighted classes
def train_lgbm_class_weight(
    X,
    y,
    *,
    num_classes=4,
    test_size=0.2,
    random_state=42,
    class_weight="balanced",
    early_stopping_rounds=50,
    log_every=50,
    lgbm_params=None,
):
    if lgbm_params is None:
        lgbm_params = {}
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    base_params = dict(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=-1,
        random_state=random_state,
    )
    if class_weight is not None:
        base_params["class_weight"] = class_weight
    base_params.update(lgbm_params)
    model = LGBMClassifier(**base_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(early_stopping_rounds),
            log_evaluation(log_every),
        ],
    )
    y_val_pred = model.predict(X_val)
    metrics = {
        "val_f1_macro": float(f1_score(y_val, y_val_pred, average="macro")),
        "val_f1_weighted": float(f1_score(y_val, y_val_pred, average="weighted")),
        "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators)),
    }
    return model, metrics, (X_val, y_val)

def lgbm_feature_importance(
    df,
    target_col="urgency_level",
    random_state=42,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=random_state,
        device="cpu",         
    )
    lgbm.fit(X, y)
    importances = lgbm.booster_.feature_importance(importance_type="gain")
    importance_df = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": importances
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["importance_norm"] = (importance_df["importance"] / importance_df["importance"].sum())
    importance_df["importance_pct"] = (round(importance_df["importance_norm"] * 100, 3))
    importance_df["importance_cum"] = importance_df["importance_pct"].cumsum()
    importance_df = importance_df.drop(columns=["importance"])
    return importance_df, lgbm

def remove_weak_features(df, var_list):
    df = df.copy()
    df.drop(columns=var_list, inplace=True, errors="ignore")
    return df

def target_summary_table(df, target_col):
    counts = df[target_col].value_counts().sort_index()
    percentages = df[target_col].value_counts(normalize=True).sort_index() * 100
    summary_df = pd.DataFrame({
        target_col: counts.index,
        "count": counts.values,
        "percentage": percentages.values.round(3)
    })
    return summary_df

def default_xgboost_cv_f1_score(
        df,
        target_col="urgency_level",
        n_folds = 10,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            n_estimators=300,   
            n_jobs=-1,  
            eval_metric="mlogloss",
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=52)
    scores = cross_val_score(
        xgb,
        X,
        y,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )
    scores_df = pd.DataFrame([{
        "mean_macro_f1": float(scores.mean()),
        "std_macro_f1": float(scores.std(ddof=1)),
        "var_macro_f1": float(scores.var(ddof=1)),
    }])
    return scores_df

def default_lgbm_cv_f1_score(
        df,
        target_col="urgency_level",
        n_folds=10,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    lgbm = LGBMClassifier(
        objective="multiclass",
        num_class=4,
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        n_jobs=-1,
        random_state=52,
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=52)
    scores = cross_val_score(
        lgbm,
        X,
        y,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )
    scores_df = pd.DataFrame([{
        "mean_macro_f1": float(scores.mean()),
        "std_macro_f1": float(scores.std(ddof=1)),
        "var_macro_f1": float(scores.var(ddof=1)),
    }])
    return scores_df

def default_sgd_cv_f1_score(
        df,
        target_col="urgency_level",
        n_folds=10,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    svm = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("svm", SGDClassifier(
            loss="log_loss",              
            class_weight="balanced",
            alpha=1e-4,               
            max_iter=2000,
            tol=1e-3,
            random_state=52,
            n_jobs=-1
        ))
    ])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=52)
    scores = cross_val_score(
        svm,
        X,
        y,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )
    scores_df = pd.DataFrame([{
        "mean_macro_f1": float(scores.mean()),
        "std_macro_f1": float(scores.std(ddof=1)),
        "var_macro_f1": float(scores.var(ddof=1)),
    }])
    return scores_df

def default_brf_cv_f1_score(
        df,
        target_col="urgency_level",
        n_folds=10,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    brf = BalancedRandomForestClassifier(
        n_estimators=300,
        random_state=52,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=52)
    scores = cross_val_score(
        brf,
        X,
        y,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )
    scores_df = pd.DataFrame([{
        "mean_macro_f1": float(scores.mean()),
        "std_macro_f1": float(scores.std(ddof=1)),
        "var_macro_f1": float(scores.var(ddof=1)),
    }])
    return scores_df

