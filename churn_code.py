import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)

# ─────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────

df = pd.read_csv('Customer Churn.csv')
df.head()

df.info()

# Replacing blanks with 0 — tenure is 0 and no total charges are recorded
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
df["TotalCharges"] = df["TotalCharges"].astype("float")

df.info()
df.isnull().sum().sum()
df.describe()
df["customerID"].duplicated().sum()

def conv(value):
    if value == 1:
        return "yes"
    else:
        return "no"

# Convert 0/1 values of SeniorCitizen to yes/no for readability
df['SeniorCitizen'] = df["SeniorCitizen"].apply(conv)


# ─────────────────────────────────────────
# 2. EDA / VISUALIZATIONS
# ─────────────────────────────────────────

ax = sns.countplot(x='Churn', data=df)
ax.bar_label(ax.containers[0])
plt.title("Count of Customers by Churn")
plt.show()

plt.figure(figsize=(3, 4))
gb = df.groupby("Churn").agg({'Churn': "count"})
plt.pie(gb['Churn'], labels=gb.index, autopct="%1.2f%%")
plt.title("Percentage of Churned Customers", fontsize=10)
plt.show()

# 26.54% of customers have churned — let's explore the reasons

plt.figure(figsize=(3, 3))
sns.countplot(x="gender", data=df, hue="Churn")
plt.title("Churn by Gender")
plt.show()

plt.figure(figsize=(4, 4))
ax = sns.countplot(x="SeniorCitizen", data=df)
ax.bar_label(ax.containers[0])
plt.title("Count of Customers by Senior Citizen")
plt.show()

total_counts = df.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True).unstack() * 100

# Plot stacked bar for SeniorCitizen vs Churn
fig, ax = plt.subplots(figsize=(4, 4))
total_counts.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x + width / 2, y + height / 2, f'{height:.1f}%', ha='center', va='center')

plt.title('Churn by Senior Citizen (Stacked Bar Chart)')
plt.xlabel('SeniorCitizen')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(title='Churn', bbox_to_anchor=(0.9, 0.9))
plt.show()

# Senior citizens churn at a comparatively higher rate

plt.figure(figsize=(9, 4))
sns.histplot(x="tenure", data=df, bins=72, hue="Churn")
plt.show()

# Long-tenure customers tend to stay; new customers (1-2 months) churn more

plt.figure(figsize=(4, 4))
ax = sns.countplot(x="Contract", data=df, hue="Churn")
ax.bar_label(ax.containers[0])
plt.title("Count of Customers by Contract")
plt.show()

# Month-to-month contract customers are far more likely to churn

df.columns.values

columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Subplot grid for service columns
n_cols = 3
n_rows = (len(columns) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(columns):
    sns.countplot(x=col, data=df, ax=axes[i], hue=df["Churn"])
    axes[i].set_title(f'Count Plot of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Customers without OnlineSecurity, TechSupport, and OnlineBackup churn more

plt.figure(figsize=(6, 4))
ax = sns.countplot(x="PaymentMethod", data=df, hue="Churn")
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
plt.title("Churned Customers by Payment Method")
plt.xticks(rotation=45)
plt.show()

# Electronic check users are more likely to churn


# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING FOR ML
# ─────────────────────────────────────────

df_model = df.drop(columns=['customerID'])

# Encode all object columns with LabelEncoder
le = LabelEncoder()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop(columns=['Churn'])
y = df_model['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ─────────────────────────────────────────
# 4. RANDOM FOREST
# ─────────────────────────────────────────

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)               # RF doesn't need scaling
y_pred_rf   = rf.predict(X_test)
y_prob_rf   = rf.predict_proba(X_test)[:, 1]


# ─────────────────────────────────────────
# 5. SVM
# ─────────────────────────────────────────

svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)       # SVM benefits from scaling
y_pred_svm  = svm.predict(X_test_scaled)
y_prob_svm  = svm.predict_proba(X_test_scaled)[:, 1]


# ─────────────────────────────────────────
# 6. MODEL COMPARISON
# ─────────────────────────────────────────

def get_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy':  round(accuracy_score(y_true, y_pred)               * 100, 2),
        'Precision': round(precision_score(y_true, y_pred)              * 100, 2),
        'Recall':    round(recall_score(y_true, y_pred)                 * 100, 2),
        'F1 Score':  round(f1_score(y_true, y_pred)                     * 100, 2),
        'ROC-AUC':   round(roc_auc_score(y_true, y_prob)                * 100, 2),
    }

metrics_rf  = get_metrics(y_test, y_pred_rf,  y_prob_rf)
metrics_svm = get_metrics(y_test, y_pred_svm, y_prob_svm)

comparison_df = pd.DataFrame({
    'Random Forest': metrics_rf,
    'SVM':           metrics_svm
})

print("\n========== Model Comparison ==========")
print(comparison_df.to_string())

# ── 6a. Bar chart comparison ──────────────────────────────────────────────────

comparison_df.plot(kind='bar', figsize=(10, 5), rot=0,
                   color=['#2ecc71', '#3498db'], edgecolor='black')

plt.title('Random Forest vs SVM — Performance Metrics (%)', fontsize=13)
plt.ylabel('Score (%)')
plt.ylim(50, 100)
plt.legend(loc='lower right')

for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.1f', padding=3, fontsize=8)

plt.tight_layout()
plt.show()

# ── 6b. Confusion matrices side-by-side ──────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, y_pred, title in zip(
    axes,
    [y_pred_rf, y_pred_svm],
    ['Random Forest', 'SVM']
):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=12)

plt.suptitle('Confusion Matrices', fontsize=14)
plt.tight_layout()
plt.show()

# ── 6c. ROC Curves ───────────────────────────────────────────────────────────

plt.figure(figsize=(7, 5))

for y_prob, label, color in [
    (y_prob_rf,  f"Random Forest (AUC={metrics_rf['ROC-AUC']}%)",  '#2ecc71'),
    (y_prob_svm, f"SVM           (AUC={metrics_svm['ROC-AUC']}%)", '#3498db'),
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=label, color=color, lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Random Forest vs SVM')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ── 6d. Feature Importance (Random Forest) ───────────────────────────────────

feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10], palette='viridis')
plt.title('Top 10 Feature Importances — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()