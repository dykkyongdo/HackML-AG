import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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
