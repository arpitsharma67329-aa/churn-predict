import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Customer Churn Analysis & Prediction")
st.markdown("Upload your dataset, explore the data, compare models, and predict churn for new customers.")

# ──────────────────────────────────────────────────────────────
# SIDEBAR — FILE UPLOAD
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Customer_Churn.csv", type=["csv"])
    st.markdown("---")
    test_size = st.slider("Test split size", 0.1, 0.4, 0.2, 0.05)
    rf_estimators = st.slider("RF: n_estimators", 50, 300, 100, 50)
    svm_kernel = st.selectbox("SVM kernel", ["rbf", "linear", "poly"])
    run_btn = st.button("🚀 Train Models", use_container_width=True)

# ──────────────────────────────────────────────────────────────
# LOAD & CLEAN
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0").astype(float)
    df["SeniorCitizen"] = df["SeniorCitizen"].apply(lambda v: "Yes" if v == 1 else "No")
    return df

@st.cache_data
def train_models(df_raw, test_sz, n_est, kernel):
    df = df_raw.copy()
    df = df.drop(columns=["customerID"])
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf  = rf.predict(X_test)
    y_prob_rf  = rf.predict_proba(X_test)[:, 1]

    svm = SVC(kernel=kernel, probability=True, random_state=42)
    svm.fit(X_train_sc, y_train)
    y_pred_svm = svm.predict(X_test_sc)
    y_prob_svm = svm.predict_proba(X_test_sc)[:, 1]

    return rf, svm, scaler, X, X_train, X_test, y_train, y_test, \
           y_pred_rf, y_prob_rf, y_pred_svm, y_prob_svm

def get_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred)  * 100, 2),
        "Precision": round(precision_score(y_true, y_pred) * 100, 2),
        "Recall":    round(recall_score(y_true, y_pred)    * 100, 2),
        "F1 Score":  round(f1_score(y_true, y_pred)        * 100, 2),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob)   * 100, 2),
    }

# ──────────────────────────────────────────────────────────────
# MAIN FLOW
# ──────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload **Customer_Churn.csv** from the sidebar to get started.")
    st.stop()

df = load_data(uploaded_file)

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 EDA", "🤖 Model Results", "🎯 Predict"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Raw Data")
    st.dataframe(df.head(50), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Churned", int(df["Churn"].value_counts().get("Yes", 0)))
    col3.metric("Churn Rate", f"{df['Churn'].value_counts(normalize=True).get('Yes',0)*100:.1f}%")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0].rename("Missing Count") if missing.any() else pd.DataFrame({"Status": ["No missing values"]}))

# ══════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Exploratory Data Analysis")

    # Row 1
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df["Churn"].value_counts()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=["#3498db", "#e74c3c"], startangle=90)
        ax.set_title("Churn Distribution")
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.countplot(x="Contract", data=df, hue="Churn", ax=ax, palette=["#3498db", "#e74c3c"])
        ax.set_title("Churn by Contract Type")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)
        plt.close()

    # Row 2
    c3, c4 = st.columns(2)

    with c3:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(x="tenure", data=df, bins=36, hue="Churn", ax=ax,
                     palette=["#3498db", "#e74c3c"])
        ax.set_title("Tenure Distribution by Churn")
        st.pyplot(fig)
        plt.close()

    with c4:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sc_churn = df.groupby("SeniorCitizen")["Churn"].value_counts(normalize=True).unstack() * 100
        sc_churn.plot(kind="bar", stacked=True, ax=ax, color=["#3498db", "#e74c3c"], edgecolor="black")
        for p in ax.patches:
            w, h = p.get_width(), p.get_height()
            x, y = p.get_xy()
            if h > 3:
                ax.text(x + w / 2, y + h / 2, f"{h:.0f}%", ha="center", va="center", fontsize=8)
        ax.set_title("Churn Rate by Senior Citizen")
        ax.set_xlabel("Senior Citizen")
        ax.set_ylabel("Percentage (%)")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Churn")
        st.pyplot(fig)
        plt.close()

    # Row 3 — Payment method
    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.countplot(x="PaymentMethod", data=df, hue="Churn", ax=ax, palette=["#3498db", "#e74c3c"])
    ax.set_title("Churn by Payment Method")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
    plt.close()

    # Row 4 — Services grid
    st.subheader("Churn by Service Type")
    service_cols = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(service_cols):
        sns.countplot(x=col, data=df, hue="Churn", ax=axes[i], palette=["#3498db", "#e74c3c"])
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(axis="x", rotation=15, labelsize=7)
        axes[i].set_xlabel("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════
with tab3:
    if not run_btn:
        st.info("Click **🚀 Train Models** in the sidebar to run the analysis.")
    else:
        with st.spinner("Training Random Forest and SVM…"):
            (rf, svm, scaler, X, X_train, X_test, y_train, y_test,
             y_pred_rf, y_prob_rf, y_pred_svm, y_prob_svm) = train_models(
                df, test_size, rf_estimators, svm_kernel
            )

        metrics_rf  = get_metrics(y_test, y_pred_rf,  y_prob_rf)
        metrics_svm = get_metrics(y_test, y_pred_svm, y_prob_svm)
        comp_df = pd.DataFrame({"Random Forest": metrics_rf, "SVM": metrics_svm})

        # Metrics table
        st.subheader("📋 Model Comparison")
        st.dataframe(comp_df.style.highlight_max(axis=1, color="#d4edda"), use_container_width=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(9, 4))
        comp_df.plot(kind="bar", ax=ax, rot=0, color=["#2ecc71", "#3498db"], edgecolor="black")
        ax.set_title("Random Forest vs SVM — Performance Metrics (%)")
        ax.set_ylabel("Score (%)")
        ax.set_ylim(50, 105)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", padding=3, fontsize=8)
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Confusion matrices
        st.subheader("Confusion Matrices")
        c1, c2 = st.columns(2)
        for col, y_pred, title in [(c1, y_pred_rf, "Random Forest"), (c2, y_pred_svm, "SVM")]:
            with col:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                cm = confusion_matrix(y_test, y_pred)
                ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(
                    ax=ax, colorbar=False, cmap="Blues"
                )
                ax.set_title(title)
                st.pyplot(fig)
                plt.close()

        # ROC curves
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for y_prob, label, color in [
            (y_prob_rf,  f"Random Forest (AUC={metrics_rf['ROC-AUC']}%)",  "#2ecc71"),
            (y_prob_svm, f"SVM           (AUC={metrics_svm['ROC-AUC']}%)", "#3498db"),
        ]:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, label=label, color=color, lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Random Forest vs SVM")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature importance
        st.subheader("Top 10 Feature Importances (Random Forest)")
        feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10], palette="viridis", ax=ax)
        ax.set_title("Top 10 Feature Importances")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Store models in session state for tab 4
        st.session_state["rf"]     = rf
        st.session_state["svm"]    = svm
        st.session_state["scaler"] = scaler
        st.session_state["X_cols"] = list(X.columns)

# ══════════════════════════════════════════════════════════════
# TAB 4 — PREDICT NEW CUSTOMER
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🎯 Predict Churn for a New Customer")

    if "rf" not in st.session_state:
        st.info("Train models first (Tab 3 → **🚀 Train Models**).")
    else:
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                gender          = st.selectbox("Gender", ["Male", "Female"])
                senior          = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner         = st.selectbox("Partner", ["Yes", "No"])
                dependents      = st.selectbox("Dependents", ["No", "Yes"])
                tenure          = st.slider("Tenure (months)", 0, 72, 12)
                phone_service   = st.selectbox("Phone Service", ["Yes", "No"])

            with c2:
                multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_sec      = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup   = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                device_prot     = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support    = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

            with c3:
                streaming_tv    = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_mov   = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment         = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
                total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)

            submitted = st.form_submit_button("Predict", use_container_width=True)

        if submitted:
            raw = {
                "gender": gender, "SeniorCitizen": senior, "Partner": partner,
                "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
                "MultipleLines": multiple_lines, "InternetService": internet,
                "OnlineSecurity": online_sec, "OnlineBackup": online_backup,
                "DeviceProtection": device_prot, "TechSupport": tech_support,
                "StreamingTV": streaming_tv, "StreamingMovies": streaming_mov,
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            input_df = pd.DataFrame([raw])
            le = LabelEncoder()
            ref_df = df.drop(columns=["customerID", "Churn"]).copy()
            ref_df["SeniorCitizen"] = ref_df["SeniorCitizen"].map({"Yes": "Yes", "No": "No"})

            for col in input_df.select_dtypes(include="object").columns:
                combined = pd.concat([ref_df[col], input_df[col]], ignore_index=True)
                le.fit(combined)
                input_df[col] = le.transform(input_df[col])

            input_df = input_df[st.session_state["X_cols"]]
            input_scaled = st.session_state["scaler"].transform(input_df)

            rf_prob  = st.session_state["rf"].predict_proba(input_df)[0][1]
            svm_prob = st.session_state["svm"].predict_proba(input_scaled)[0][1]
            avg_prob = (rf_prob + svm_prob) / 2

            st.markdown("---")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("🌲 Random Forest", f"{rf_prob*100:.1f}%", help="Churn probability")
            mc2.metric("🔵 SVM",           f"{svm_prob*100:.1f}%", help="Churn probability")
            mc3.metric("⚖️ Ensemble Avg",  f"{avg_prob*100:.1f}%", help="Average of both models")

            if avg_prob >= 0.5:
                st.error(f"⚠️ **High churn risk** — this customer is likely to churn ({avg_prob*100:.1f}% probability).")
            else:
                st.success(f"✅ **Low churn risk** — this customer is likely to stay ({(1-avg_prob)*100:.1f}% retention probability).")
                import streamlit as st

st.title("Churn Prediction App 🚀")
st.write("Deployment Successful on Railway!")

