TARGET_COL = "urgency_level"

CORRELATED_FEATURES_TO_REMOVE = ["id","newbalanceOrig", "newbalanceDest"]

LOG_TRANSFORM_FEATURES = ["amount", "oldbalanceOrg", "oldbalanceDest"]

NAME_COLS = ["nameOrig", "nameDest"]

CATEGORICAL_COL = "type"

CLASS_WEIGHT = {0: 1.0, 1: 3.0, 2: 2.0, 3: 2.0}

WEAK_FEATURES_TO_REMOVE = ["nameOrig", "nameDest", "type_DEBIT"]