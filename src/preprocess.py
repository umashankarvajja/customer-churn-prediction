import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    
    # Drop customer ID (not useful for prediction)
    df.drop(columns=["customerID"], inplace=True)
    
    # Fix TotalCharges column (has spaces instead of numbers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    
    # Convert target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    # Encode all categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

def split_features(df, target="Churn"):
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y
