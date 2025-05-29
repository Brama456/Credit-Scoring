import pandas as pd
import joblib
import os
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import lime
import lime.lime_tabular

# ---------------- STEP 1: Load and preprocess data ----------------
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)

    # Label encode the target (CreditRating)
    le = LabelEncoder()
    df['CreditRating'] = le.fit_transform(df['CreditRating'])

    # One-hot encode categorical features (except target)
    X = pd.get_dummies(df.drop('CreditRating', axis=1))
    y = df['CreditRating']

    return X, y, le

# ---------------- STEP 2: Train XGBoost model ----------------
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)

    # Predictions & evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n🔍 Accuracy: {acc:.2f}")
    print("\n📊 Classification Report:\n", report)
    print("\n🧩 Confusion Matrix:\n", cm)

    # Save model & encoder
    joblib.dump(model, 'xgb_credit_model.pkl')
    return model, X_train.values, y_train.values

# ---------------- STEP 3: Streamlit interface ----------------
def run_streamlit_app(model, le, feature_columns, X_train_np):
    st.title("💳 Credit Rating Predictor")

    st.write("### Enter the details below:")

    user_input = {}
    for col in feature_columns:
        user_input[col] = st.text_input(f"{col}", value="0")

    if st.button("Predict Credit Rating"):
        input_df = pd.DataFrame([user_input])

        # Convert to numeric
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # One-hot encode & align
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        pred_encoded = model.predict(input_df)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]

        st.success(f"🌟 Predicted Credit Rating: **{pred_label}**")

        # --- LIME explanation ---
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_np,
            feature_names=feature_columns,
            class_names=le.classes_,
            discretize_continuous=True,
            mode='classification'
        )

        explanation = explainer.explain_instance(
            input_df.iloc[0].values,
            model.predict_proba,
            num_features=10
        )

        st.write("### Explanation of the prediction (LIME):")
        lime_html = explanation.as_html()
        st.components.v1.html(lime_html, height=400, scrolling=True)

# ---------------- MAIN ----------------
def main():
    excel_file = 'Credit score dataset.xlsx'

    if os.path.exists('xgb_credit_model.pkl') and os.path.exists('label_encoder.pkl'):
        st.write("✅ Loading saved model...")
        model = joblib.load('xgb_credit_model.pkl')
        le = joblib.load('label_encoder.pkl')

        # Load data just to get feature columns
        X, y, _ = load_and_preprocess_data(excel_file)
        X_train_np, _, _, _ = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42
        )
    else:
        st.write("🔨 Training model...")
        X, y, le = load_and_preprocess_data(excel_file)
        model, X_train_np, _ = train_and_evaluate(X, y)
        joblib.dump(le, 'label_encoder.pkl')

    run_streamlit_app(model, le, X.columns, X_train_np)

if __name__ == "__main__":
    main()
