
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="CrediFast - Risco de Crédito", layout="wide")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# Funções
# -----------------------------
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def preprocess(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "loan_status"]
    cat_cols = [c for c in df.columns if c not in num_cols + ["loan_status"]]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    return preprocessor, X_train_prep, y_train, X_test_prep, y_test, num_cols, cat_cols

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    auc = roc_auc_score(y_test, y_score)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_title(f'ROC - {name}')
    ax.legend()

    return {"name": name, "auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, fig

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Upload do CSV")
uploaded = st.sidebar.file_uploader("Envie o arquivo credit_risk_dataset.csv", type=["csv"])
df = load_data(uploaded)

if df is None:
    st.warning("Envie o arquivo CSV para continuar.")
    st.stop()

subset_frac = 0.3  # para clusters/outliers

# -----------------------------
# Layout principal
# -----------------------------
st.title("CrediFast • APP Rápido e Completo")
tabs = st.tabs(["Dados", "Modelos", "Explicabilidade", "Clusters", "Outliers", "Recomendações"])

# -----------------------------
# Aba: Dados
# -----------------------------
with tabs[0]:
    st.subheader("Amostra dos dados")
    st.dataframe(df.head(20))
    fig, ax = plt.subplots(figsize=(5,3))
    sns.countplot(x="loan_status", data=df, ax=ax)
    ax.set_title("Distribuição da variável alvo")
    st.pyplot(fig)

# -----------------------------
# Pré-processamento
# -----------------------------
preprocessor, X_train, y_train, X_test, y_test, num_cols, cat_cols = preprocess(df)

# -----------------------------
# Aba: Modelos
# -----------------------------
with tabs[1]:
    st.subheader("Treinar modelos")
    if st.button("▶️ Treinar"):
        models = [
            ("DecisionTree", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
            ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)),
            ("SVM", SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
        ]
        results = []
        cols = st.columns(3)
        i = 0
        for name, model in models:
            res, fig = evaluate_model(name, model, X_train, y_train, X_test, y_test)
            results.append(res)
            with cols[i % 3]:
                st.pyplot(fig)
            i += 1
        st.dataframe(pd.DataFrame(results).sort_values(by=["auc", "recall"], ascending=[False, False]))

# -----------------------------
# Aba: Explicabilidade
# -----------------------------
with tabs[2]:
    st.subheader("Explicabilidade (Permutation Importance)")
    best_model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE).fit(X_train, y_train)
    pi = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE)
    feature_names = num_cols + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))
    import_df = pd.DataFrame({"feature": feature_names, "importance": pi.importances_mean}).sort_values(by="importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(y="feature", x="importance", data=import_df.head(15), ax=ax)
    ax.set_title("Top 15 - Permutation Importance")
    st.pyplot(fig)

# -----------------------------
# Aba: Clusters
# -----------------------------
with tabs[3]:
    st.subheader("Clusters (KMeans + PCA)")
    df_sub = df.sample(frac=subset_frac, random_state=RANDOM_STATE)
    X_scaled = RobustScaler().fit_transform(preprocessor.transform(df_sub.drop(columns=["loan_status"])))
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE).fit(X_pca)
    clusters = kmeans.labels_
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="tab10", ax=ax)
    st.pyplot(fig)

# -----------------------------
# Aba: Outliers
# -----------------------------
with tabs[4]:
    st.subheader("Outliers (DBSCAN)")
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X_scaled)
    distances, _ = neigh.kneighbors(X_scaled)
    eps_val = np.percentile(np.sort(distances[:, -1]), 95)
    db = DBSCAN(eps=eps_val, min_samples=5).fit(X_scaled)
    outliers = (db.labels_ == -1).sum()
    st.write(f"Outliers detectados: {outliers}")

# -----------------------------
# Aba: Recomendações
# -----------------------------
with tabs[5]:
    st.write("Clientes com parcela/renda alta e juros elevados devem ter limites reduzidos ou exigência de garantias. "
             "Clusters com maior inadimplência e outliers devem passar por análise reforçada. "
             "Monitorar continuamente métricas e ajustar políticas conforme padrões explicativos.")
