
# ============================================
# CrediFast - APP de Risco de Crédito (Do Zero, Completo + Rápido)
# - Modo Rápido (default) e Modo Completo (opcional)
# - Fallbacks: SHAP/XGB/LightGBM/SMOTE -> nunca quebra nem perde conteúdo
# - Cache, botão de treino e subset para performance
# ============================================

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Núcleo SciKit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

# Importes opcionais com fallback
HAS_SMOTE = True
try:
    from imbalanced_learn.over_sampling import SMOTE
except Exception:
    HAS_SMOTE = False

HAS_XGB = True
try:
    import xgboost as xgb
except Exception:
    HAS_XGB = False

HAS_LGBM = True
try:
    import lightgbm as lgb
except Exception:
    HAS_LGBM = False

HAS_SHAP = True
try:
    import shap
except Exception:
    HAS_SHAP = False

import joblib

# -----------------------------
# Configurações gerais
# -----------------------------
st.set_page_config(page_title="CrediFast - Risco de Crédito", layout="wide")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_context("talk")

# Avisos de ambiente (uma única vez)
if not HAS_SMOTE:
    st.caption("⚠️ SMOTE indisponível (imbalanced-learn). O app funciona; você pode treinar sem balanceamento.")
if not HAS_XGB:
    st.caption("⚠️ XGBoost indisponível. O Modo Completo funcionará sem XGBoost.")
if not HAS_LGBM:
    st.caption("⚠️ LightGBM indisponível. O Modo Completo funcionará sem LightGBM.")
if not HAS_SHAP:
    st.caption("⚠️ SHAP indisponível. A explicabilidade usa Permutation Importance (fallback).")

# -----------------------------
# Utilitários & Cache
# -----------------------------
@st.cache_data
def load_data(uploaded_file=None, local_path="credit_risk_dataset.csv"):
    """Carrega o CSV do upload ou da raiz do repositório."""
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        return None
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

def build_preprocessor(df, target_col="loan_status"):
    """Cria ColumnTransformer com imputação/escala/one-hot, robusto a versões do sklearn."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    cat_cols = [c for c in df.columns if c not in num_cols + [target_col]]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # OneHotEncoder: usar sparse_output=False se disponível; senão, cair para sparse=False
    try:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
    except TypeError:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor, num_cols, cat_cols

@st.cache_data
def split_transform_balance(df, target_col="loan_status", use_smote=False):
    """Split estratificado, fit/preprocess no treino, transform no teste e SMOTE opcional."""
    preprocessor, num_cols, cat_cols = build_preprocessor(df, target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep  = preprocessor.transform(X_test)

    if use_smote and HAS_SMOTE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_bal, y_train_bal = sm.fit_resample(X_train_prep, y_train)
    else:
        X_train_bal, y_train_bal = X_train_prep, y_train

    return preprocessor, num_cols, cat_cols, X_train_bal, y_train_bal, X_test_prep, y_test

def evaluate_model(name, model, X_train, y_train, X_test, y_test, plot_roc=True):
    """Treina, prevê, calcula métricas e (opcionalmente) curva ROC."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        try:
            y_score = model.decision_function(X_test)
            y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-8)
        except Exception:
            y_score = y_pred

    auc = roc_auc_score(y_test, y_score)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    fig = None
    if plot_roc:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        ax.plot([0,1], [0,1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Recall)')
        ax.set_title(f'ROC - {name}')
        ax.legend()

    return {"name": name, "model": model, "auc": auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, fig

@st.cache_data
def cached_evaluate(name, model, X_train, y_train, X_test, y_test):
    """Cache para não recalcular sempre quando muda de aba."""
    return evaluate_model(name, model, X_train, y_train, X_test, y_test, plot_roc=True)

def get_feature_names(preprocessor, num_cols, cat_cols):
    """Nomes pós-OneHot para usar em explicabilidade."""
    onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
    try:
        cat_names = list(onehot.get_feature_names_out(cat_cols))
    except Exception:
        # fallback se versão antiga não tem get_feature_names_out
        cat_names = [f"{c}" for c in cat_cols]
    return num_cols + cat_names

# -----------------------------
# Sidebar - Upload & Configuração
# -----------------------------
st.sidebar.title("📥 Upload & Configurações")
uploaded = st.sidebar.file_uploader("Envie o arquivo CSV (credit_risk_dataset.csv)", type=["csv"])
df = load_data(uploaded)

if df is None:
    st.warning("Nenhum CSV encontrado. Envie o dataset para continuar.")
    st.stop()

mode = st.sidebar.radio("Modo do app", ["Rápido (recomendado)", "Completo (tudo)"], index=0)
use_smote = st.sidebar.checkbox("Usar SMOTE no treino (balancear)", value=False and HAS_SMOTE)
n_estimators = st.sidebar.selectbox("n_estimators árvores/boosting", [50, 100, 150, 200], index=1)
shap_sample = st.sidebar.selectbox("Amostra SHAP (summary)", [200, 300, 500, 800], index=0)
subset_frac = st.sidebar.slider("Fração para PCA/KMeans/DBSCAN", 0.2, 1.0, 0.5, 0.1)

# -----------------------------
# Layout principal
# -----------------------------
st.title("CrediFast • Sistema de Apoio à Decisão de Risco de Crédito")
tabs = st.tabs(["Dados", "Modelos & Métricas", "Explicabilidade", "Clusters (PCA+KMeans)", "Outliers (DBSCAN)", "Recomendações"])

# -----------------------------
# Aba: Dados
# -----------------------------
with tabs[0]:
    st.subheader("Amostra dos dados")
    st.dataframe(df.head(20), use_container_width=True)

    if "loan_status" in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x="loan_status", data=df, ax=ax)
        ax.set_title("Distribuição da variável-alvo (loan_status)")
        ax.set_xticklabels(["Good (0)", "Bad (1)"])
        st.pyplot(fig)

        num_cols_tmp = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols_tmp = [c for c in num_cols_tmp if c != "loan_status"]
        if len(num_cols_tmp) > 0:
            corrs = df[num_cols_tmp + ["loan_status"]].corr()["loan_status"].drop("loan_status").abs().sort_values(ascending=False)
            st.write("Top correlações absolutas com loan_status:")
            st.write(corrs.head(10))

# -----------------------------
# Pré-processamento (cache)
# -----------------------------
t0 = time.time()
preprocessor, num_cols, cat_cols, X_train_bal, y_train_bal, X_test_prep, y_test = split_transform_balance(
    df, target_col="loan_status", use_smote=use_smote
)
st.caption(f"🔧 Pré-processamento concluído em {time.time()-t0:.1f}s.")

# -----------------------------
# Aba: Modelos & Métricas (com botão)
# -----------------------------
with tabs[1]:
    st.subheader("Treinamento e avaliação de modelos")
    start_train = st.button("▶️ Treinar/Atualizar")
    if not start_train:
        st.info("Clique no botão para iniciar o treino (evita processamento automático).")
        st.stop()

    # MODELOS conforme modo
    models = []
    if mode.startswith("Rápido"):
        models = [
            ("DecisionTree", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
            ("RandomForest", RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE)),
            ("SVM", SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)),
        ]
    else:
        # Completo: inclui todo o conjunto pedido na prova
        models = [
            ("KNN", None),  # será definido abaixo
            ("SVM", SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)),
            ("DecisionTree", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
            ("RandomForest", RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=RANDOM_STATE)),
            ("AdaBoost", AdaBoostClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE)),
            ("GradientBoosting", GradientBoostingClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE)),
            ("MLP", MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300, random_state=RANDOM_STATE)),
        ]
        # KNN (com dados já escalados no preprocessor)
        from sklearn.neighbors import KNeighborsClassifier
        models[0] = ("KNN", KNeighborsClassifier(n_neighbors=7))
        if HAS_XGB:
            models.append(("XGBoost", xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='logloss', n_estimators=n_estimators, n_jobs=-1, random_state=RANDOM_STATE
            )))
        if HAS_LGBM:
            models.append(("LightGBM", lgb.LGBMClassifier(
                n_estimators=n_estimators, n_jobs=-1, random_state=RANDOM_STATE
            )))

    # Treino + avaliação
    results = []
    cols = st.columns(3)
    i_plot = 0
    t1 = time.time()
    for name, model in models:
        res, fig = cached_evaluate(name, model, X_train_bal, y_train_bal, X_test_prep, y_test)
        results.append(res)
        if fig is not None:
            with cols[i_plot % 3]:
                st.pyplot(fig)
            i_plot += 1
    st.caption(f"⏱️ Treino+avaliação: {time.time()-t1:.1f}s")

    # Tabela + melhor
    results_df = pd.DataFrame(results).sort_values(by=["auc", "recall"], ascending=[False, False]).reset_index(drop=True)
    st.dataframe(results_df, use_container_width=True)

    best_row = results_df.iloc[0]
    best_name = best_row["name"]
    best_model = None
    # reencontrar o objeto do modelo pelo nome
    for i, (nm, mdl) in enumerate(models):
        if nm == best_name:
            best_model = mdl
            break
    st.success(f"🟢 Modelo vencedor: **{best_name}** (AUC={best_row['auc']:.3f} | Recall={best_row['recall']:.3f})")

    # Persistência & download (opcional)
    if st.checkbox("Salvar artefatos (preprocessor + best_model)"):
        joblib.dump(preprocessor, "preprocessor.joblib")
        joblib.dump(best_model, f"best_model_{best_name}.joblib")
        st.info("Salvos: preprocessor.joblib e best_model_*.joblib")
        if os.path.exists("preprocessor.joblib"):
            with open("preprocessor.joblib", "rb") as f:
                st.download_button("⬇️ Baixar preprocessor.joblib", data=f, file_name="preprocessor.joblib")
        if os.path.exists(f"best_model_{best_name}.joblib"):
            with open(f"best_model_{best_name}.joblib", "rb") as f:
                st.download_button(f"⬇️ Baixar best_model_{best_name}.joblib", data=f, file_name=f"best_model_{best_name}.joblib")

# -----------------------------
# Aba: Explicabilidade (SHAP ou Permutation Importance)
# -----------------------------
with tabs[2]:
    st.subheader("Explicabilidade do modelo vencedor")
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)

    can_tree_explain = best_name in ["RandomForest", "GradientBoosting", "DecisionTree"] + (["XGBoost", "LightGBM"] if HAS_XGB or HAS_LGBM else [])
    if HAS_SHAP and can_tree_explain:
        # SHAP (TreeExplainer) — sample control
        try:
            explainer = shap.TreeExplainer(best_model)
            sample_idx = np.random.choice(np.arange(X_test_prep.shape[0]), size=min(shap_sample, X_test_prep.shape[0]), replace=False)
            X_shap = X_test_prep[sample_idx]
            shap_values = explainer.shap_values(X_shap)
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

            st.write("**Summary plot (impacto global):**")
            fig = plt.figure(figsize=(9,7))
            shap.summary_plot(shap_vals, X_shap, feature_names=feature_names, show=False)
            st.pyplot(fig)

            # Local: um bad e um good
            y_test_arr = y_test.values
            idx_bad = np.where(y_test_arr == 1)[0]
            idx_good = np.where(y_test_arr == 0)[0]
            if len(idx_bad) and len(idx_good):
                i_bad = int(idx_bad[0]); i_good = int(idx_good[0])
                X_local = X_test_prep[[i_bad, i_good]]
                shap_values_local = explainer.shap_values(X_local)
                shap_vals_local = shap_values_local[1] if isinstance(shap_values_local, list) else shap_values_local
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

                st.write("**Waterfall - caso bad**")
                fig1 = shap.plots.waterfall(shap.Explanation(values=shap_vals_local[0], base_values=base_val, data=X_local[0], feature_names=feature_names), show=False)
                st.pyplot(fig1)

                st.write("**Waterfall - caso good**")
                fig2 = shap.plots.waterfall(shap.Explanation(values=shap_vals_local[1], base_values=base_val, data=X_local[1], feature_names=feature_names), show=False)
                st.pyplot(fig2)
        except Exception as e:
            st.error(f"Erro SHAP: {e}")
            st.info("Usando fallback (Permutation Importance) abaixo.")

    else:
        # Fallback: Permutation Importance (nunca deixa sem explicabilidade)
        st.write("**Permutation Importance (fallback de explicabilidade):**")
        try:
            pi = permutation_importance(best_model, X_test_prep, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
            import_df = pd.DataFrame({
                "feature": feature_names,
                "importance_mean": pi.importances_mean,
                "importance_std": pi.importances_std
            }).sort_values(by="importance_mean", ascending=False)
            fig, ax = plt.subplots(figsize=(9,6))
            sns.barplot(y="feature", x="importance_mean", data=import_df.head(20), ax=ax)
            ax.set_title("Top 20 - Permutation Importance")
            st.pyplot(fig)
            st.write(import_df.head(30))
        except Exception as e:
            st.error(f"Erro no fallback de explicabilidade: {e}")

# -----------------------------
# Aba: Clusters (PCA + KMeans)
# -----------------------------
with tabs[3]:
    st.subheader("Clusters (KMeans) com visualização PCA")
    try:
        df_sub = df.sample(frac=subset_frac, random_state=RANDOM_STATE) if 0 < subset_frac < 1.0 else df.copy()
        X_all = df_sub.drop(columns=["loan_status"])
        X_prep_all = preprocessor.transform(X_all)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_prep_all)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)

        from sklearn.metrics import silhouette_score
        best_k, best_score = 2, -1.0
        for k in range(2, 7):
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X_pca)
            sc = silhouette_score(X_pca, labels)
            if sc > best_score:
                best_k, best_score = k, sc

        st.write(f"Melhor k (silhouette): **{best_k}** | score={best_score:.3f} | subset={subset_frac:.2f}")

        km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
        clusters = km.fit_predict(X_pca)

        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="tab10", ax=ax, s=20, legend="full")
        ax.set_title(f"KMeans (k={best_k}) sobre PCA(2)")
        st.pyplot(fig)

        clusters_series = pd.Series(clusters, index=df_sub.index, name="cluster")
        cluster_risk = pd.concat([df_sub["loan_status"], clusters_series], axis=1).groupby("cluster")["loan_status"].agg(["count", "sum"])
        cluster_risk["bad_rate"] = cluster_risk["sum"] / cluster_risk["count"]
        st.write("Bad rate por cluster (subset):")
        st.dataframe(cluster_risk.sort_values("bad_rate", ascending=False))
    except Exception as e:
        st.error(f"Erro em clusters: {e}")

# -----------------------------
# Aba: Outliers (DBSCAN)
# -----------------------------
with tabs[4]:
    st.subheader("Outliers (DBSCAN)")
    try:
        df_sub = df.sample(frac=subset_frac, random_state=RANDOM_STATE) if 0 < subset_frac < 1.0 else df.copy()
        X_all = df_sub.drop(columns=["loan_status"])
        X_prep_all = preprocessor.transform(X_all)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_prep_all)

        neigh = NearestNeighbors(n_neighbors=5)
        nbrs = neigh.fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        kdist = np.sort(distances[:, -1])

        fig, ax = plt.subplots(figsize=(7,3))
        ax.plot(kdist)
        ax.set_title("k-distance (k=5) - escolha visual do eps (subset)")
        st.pyplot(fig)

        eps_val = float(np.percentile(kdist, 95))
        st.write(f"eps (p95): **{eps_val:.4f}** | subset={subset_frac:.2f}")

        db = DBSCAN(eps=eps_val, min_samples=5)
        labels = db.fit_predict(X_scaled)
        out_mask = (labels == -1)

        df_out = df_sub.copy()
        df_out["is_outlier"] = out_mask.astype(int)
        summary_out = df_out.groupby("is_outlier")["loan_status"].agg(["count", "sum"])
        summary_out["bad_rate"] = summary_out["sum"] / summary_out["count"]
        st.write("Inliers vs Outliers (subset):")
        st.dataframe(summary_out)
    except Exception as e:
        st.error(f"Erro em DBSCAN: {e}")

# -----------------------------
# Aba: Recomendações
# -----------------------------
with tabs[5]:
    st.subheader("Recomendações gerenciais")
    st.write(
        "Priorizar recall no limiar do modelo vencedor para reduzir falsos negativos (perdas). "
        "Parcela/renda alta (loan_percent_income) e juros altos (loan_int_rate) sugerem limites mais conservadores, "
        "entrada maior ou verificações adicionais. Renda elevada e histórico de crédito longo tendem a reduzir o risco, "
        "viabilizando condições padrão. Clusters com maior bad_rate e casos outliers devem ir para esteira reforçada "
        "(documentação extra, referências) e monitoramento proativo. Revisar continuamente AUC/Recall/Precision, ajustar "
        "o limiar e atualizar políticas conforme padrões observados em explicabilidade e nas taxas por segmento."
    )
