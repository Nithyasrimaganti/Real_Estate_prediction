
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import io
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="Real Estate Prediction App", layout="wide")

st.title("Real Estate Prediction App ")
st.markdown(
    """
This app is useful for real estate prediction and contains. 
**Sections**:
- Dataset overview & EDA 
- Feature engineering pipeline 
- Train models (classification & regression) 
- Prediction form: enter property details and get predictions
"""
)

# Load CSV automatically from Google Drive
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1eTe0FFL86mXBLQBHrrY5VEnNCM0X6uwk"
    return pd.read_csv(url)

df_raw = load_data()
st.success("Dataset loaded successfully from Google Drive!")

st.header("1) Dataset preview")
st.write("Rows:", df_raw.shape[0], " Columns:", df_raw.shape[1])
st.dataframe(df_raw.head())

# Basic EDA stats
st.subheader("Basic statistics")
st.write(df_raw.describe(include='all').T)

# === Feature engineering function ===
def feature_engineer(df_raw):
    df = df_raw.copy()
    # Price per sqft (ensure no division by zero)
    df["Price_per_Sqft"] = df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"].replace(0, np.nan)
    df["School_Density_Score"] = df["Nearby_Schools"] / df["Size_in_SqFt"].replace(0, np.nan)
    df["Hospital_Density_Score"] = df["Nearby_Hospitals"] / df["Size_in_SqFt"].replace(0, np.nan)

    # Age categories
    age_bins = [0,5,10,20,50,100]
    age_labels = ["0-5 yrs","5-10 yrs","10-20 yrs","20-50 yrs","50+ yrs"]
    df["Property_Age_Category"] = pd.cut(df["Age_of_Property"], bins=age_bins, labels=age_labels)

    # Public transport mapping
    if "Public_Transport_Accessibility" in df.columns:
        df["Public_Transport_Accessibility"] = df["Public_Transport_Accessibility"].map({"Low":1,"Medium":2,"High":3})

    # Yes/No -> 1/0
    df.replace({"Yes": 1, "No": 0}, inplace=True)
    df = df.infer_objects(copy=False)

    # Map age category to int
    age_mapping = {"0-5 yrs":1,"5-10 yrs":2,"10-20 yrs":3,"20-50 yrs":4,"50+ yrs":5}
    df["Property_Age_Category"] = df["Property_Age_Category"].map(age_mapping)

    # Amenities one-hot (if column exists)
    if "Amenities" in df.columns:
        df["Amenities"] = df["Amenities"].fillna("").astype(str).apply(lambda x: [i.strip() for i in x.split(",") if i.strip()!=""])
        all_amenities = sorted({item for sublist in df["Amenities"] for item in sublist})
        for amenity in all_amenities:
            df[f"Amenities_{amenity}"] = df["Amenities"].apply(lambda x: 1 if amenity in x else 0)
        df.drop(columns=["Amenities"], inplace=True)

    # One-hot encoding for categorical columns (safe subset)
    categorical_cols = [c for c in ["State","City","Locality","Property_Type","Furnished_Status","Facing","Owner_Type","Availability_Status"] if c in df.columns]
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc = encoder.fit_transform(df[categorical_cols])
        enc_df = pd.DataFrame(enc, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
        df = pd.concat([df.drop(columns=categorical_cols), enc_df], axis=1)

    # Fill numeric na with median
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Create targets (based on raw values per your notebook)
    r = 0.08; t = 5
    df["Future_Price_5Y"] = df_raw["Price_in_Lakhs"] * ((1+r)**t)
    price_median = df_raw["Price_in_Lakhs"].median()
    pps_median = df_raw["Price_per_SqFt"].median() if "Price_per_SqFt" in df_raw.columns else (df["Price_in_Lakhs"].median())
    rule_price = df_raw["Price_in_Lakhs"] <= price_median
    rule_pps = df_raw["Price_per_SqFt"] <= pps_median if "Price_per_SqFt" in df_raw.columns else pd.Series([False]*len(df_raw))
    rule_bhk = df_raw["BHK"] >= 3 if "BHK" in df_raw.columns else pd.Series([False]*len(df_raw))
    df["Good_Investment"] = ((rule_price.astype(int) + rule_pps.astype(int) + rule_bhk.astype(int)) >= 2).astype(int)

    return df

with st.spinner("Running feature engineering..."):
    df = feature_engineer(df_raw)

st.success("Feature engineering complete")
st.write("Processed dataset shape:", df.shape)
st.dataframe(df.head())

# === Prepare features and targets ===
drop_cols = [c for c in ["Good_Investment","Future_Price_5Y"] if c in df.columns]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y_cls = df["Good_Investment"] if "Good_Investment" in df.columns else None
y_reg = df["Future_Price_5Y"] if "Future_Price_5Y" in df.columns else None
# FIX: Convert category dtype → int for XGBoost
for col in X.select_dtypes(include=["category"]).columns:
    X[col] = X[col].astype(int)

# Also fix specifically Property_Age_Category
if "Property_Age_Category" in X.columns:
    X["Property_Age_Category"] = X["Property_Age_Category"].astype(int)


# Scale numeric columns
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
scaler = StandardScaler()
if len(numeric_cols) > 0:
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train models section
st.header("2) Train models (press the button to train)")
col1, col2 = st.columns(2)
with col1:
    train_cls = st.button("Train Classification Model (Logistic Regression)")
with col2:
    train_reg = st.button("Train Regression Model (XGBoost)")

# To store trained models
if "clf_model" not in st.session_state:
    st.session_state["clf_model"] = None
if "xgb_model" not in st.session_state:
    st.session_state["xgb_model"] = None
if "feature_columns" not in st.session_state:
    st.session_state["feature_columns"] = feature_cols

def train_classification(X, y):
    if y is None:
        st.error("Classification target not found in dataset.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("ROC AUC:", roc_auc_score(y_test, y_prob))
    st.text(classification_report(y_test, y_pred))
    return clf
def train_regression(X, y):
    # FIX for XGBoost categorical error
    X = X.copy()
    for col in X.select_dtypes(include=["category"]).columns:
        X[col] = X[col].astype(int)

    # Now safe to split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

def train_regression(X, y):
    if y is None:
        st.error("Regression target not found in dataset.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb = XGBRegressor(n_estimators=250, max_depth=8, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("RMSE:", rmse)
    st.write("MAE:", mae)
    st.write("R2:", r2)
    return xgb

if train_cls:
    with st.spinner("Training classification model..."):
        st.session_state["clf_model"] = train_classification(X.copy(), y_cls)
        st.success("Classification model trained and stored in session")

if train_reg:
    with st.spinner("Training regression model..."):
        st.session_state["xgb_model"] = train_regression(X.copy(), y_reg)
        st.success("Regression model trained and stored in session")

# Prediction form
st.header("3) Prediction — Enter property details to get predictions")
with st.form("predict_form"):
    st.write("Fill the available fields (use default values if unsure)")
    # Show a subset of numeric fields for input if present in original df_raw
    inputs = {}
    # Numeric inputs
    if "Size_in_SqFt" in df_raw.columns:
        inputs["Size_in_SqFt"] = st.number_input("Size (SqFt)", value=float(df_raw["Size_in_SqFt"].median()))
    if "Price_in_Lakhs" in df_raw.columns:
        inputs["Price_in_Lakhs"] = st.number_input("Current Price (Lakhs)", value=float(df_raw["Price_in_Lakhs"].median()))
    if "Nearby_Schools" in df_raw.columns:
        inputs["Nearby_Schools"] = st.number_input("Nearby Schools", value=int(df_raw["Nearby_Schools"].median()))
    if "Nearby_Hospitals" in df_raw.columns:
        inputs["Nearby_Hospitals"] = st.number_input("Nearby Hospitals", value=int(df_raw["Nearby_Hospitals"].median()))
    if "BHK" in df_raw.columns:
        inputs["BHK"] = st.number_input("BHK", value=int(df_raw["BHK"].median()))
    # Categorical simplified options
    if "Public_Transport_Accessibility" in df_raw.columns:
        pta = st.selectbox("Public Transport (Low/Medium/High)", options=["Low","Medium","High"])
        inputs["Public_Transport_Accessibility"] = 1 if pta=="Low" else (2 if pta=="Medium" else 3)
    submit = st.form_submit_button("Predict")
    
if submit:
    # Build a single-row dataframe with all feature columns used in training
    row = {c: 0 for c in st.session_state["feature_columns"]}
    # Fill values we know from inputs; others remain 0
    for k,v in inputs.items():
        if k in row:
            row[k] = v
        else:
            # handle common engineered features
            if k=="Size_in_SqFt":
                row["Size_in_SqFt"] = v
            if k=="Price_in_Lakhs":
                row["Price_in_Lakhs"] = v
            if k=="Nearby_Schools":
                row["Nearby_Schools"] = v
            if k=="Nearby_Hospitals":
                row["Nearby_Hospitals"] = v
            if k=="BHK":
                row["BHK"] = v
            if k=="Public_Transport_Accessibility":
                row["Public_Transport_Accessibility"] = v

    X_new = pd.DataFrame([row], columns=st.session_state["feature_columns"])
    # Ensure numeric scaling (use earlier scaler logic: scale numeric cols only)
    numeric_cols = X_new.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) > 0:
        try:
            X_new[numeric_cols] = scaler.transform(X_new[numeric_cols])
        except Exception:
            # fallback: no scaler available -> pass
            pass

    st.subheader("Prediction Results")
    if st.session_state.get("clf_model") is not None:
        cls_pred = st.session_state["clf_model"].predict(X_new)[0]
        cls_prob = st.session_state["clf_model"].predict_proba(X_new)[0][1] if hasattr(st.session_state["clf_model"], "predict_proba") else None
        st.write("Good Investment (classification):", int(cls_pred))
        if cls_prob is not None:
            st.write("Probability:", float(cls_prob))
    else:
        st.info("Classification model not trained. Click 'Train Classification Model' to train it.")

    if st.session_state.get("xgb_model") is not None:
        reg_pred = st.session_state["xgb_model"].predict(X_new)[0]
        st.write("Predicted Future Price (5Y) in Lakhs:", float(reg_pred))
    else:
        st.info("Regression model not trained. Click 'Train Regression Model' to train it.")

# Save models and columns
st.header("4) Save models & artifacts (optional)")
if st.button("Save trained models to disk"):
    if st.session_state.get("clf_model") is not None:
        joblib.dump(st.session_state["clf_model"], "clf_model.joblib")
    if st.session_state.get("xgb_model") is not None:
        joblib.dump(st.session_state["xgb_model"], "xgb_model.joblib")
    joblib.dump(st.session_state["feature_columns"], "model_columns.pkl")
    st.success("Saved models and model_columns.pkl to the working directory.")

st.markdown("---")

