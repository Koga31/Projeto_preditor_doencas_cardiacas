"""
Classificação de risco de doença cardíaca a partir de 11 atributos clínicos.

Pipeline:
  1. Limpeza: valores 0 em Colesterol e Pressão em repouso são registros faltantes
     (fisiologicamente impossíveis) e são tratados como NaN -> imputação pela mediana.
  2. Pré-processamento: One-Hot Encoding nas variáveis categóricas e padronização das numéricas.
  3. Modelos: Regressão Logística (baseline interpretável) x Random Forest.
  4. Avaliação: validação cruzada estratificada (5 folds) + conjunto de teste separado.
     Em triagem médica o RECALL da classe positiva é a métrica principal:
     um falso negativo (doente classificado como saudável) é o erro mais caro.

Uso:
    python heart_failure_predictor.py     # treina, avalia e salva os gráficos em images/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "heart.csv"
IMAGES_DIR = BASE_DIR / "images"
RANDOM_STATE = 31
TARGET = "HeartDisease"


def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    # Zeros impossíveis = dado ausente no registro original
    for col in ("Cholesterol", "RestingBP"):
        df[col] = df[col].replace(0, np.nan)
    return df


def montar_preprocessador(X: pd.DataFrame) -> ColumnTransformer:
    categoricas = X.select_dtypes(exclude="number").columns.tolist()
    numericas = X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ]
    )


def main():
    df = carregar_dados()
    print(f"Pacientes: {len(df)} | com doença: {df[TARGET].mean():.1%}")
    print(f"Colesterol ausente (antes registrado como 0): {df['Cholesterol'].isna().sum()} registros\n")

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    linhas, pipes = [], {}
    for nome, modelo in modelos.items():
        pipe = Pipeline([("prep", montar_preprocessador(X)), ("modelo", modelo)])
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["accuracy", "recall", "roc_auc"])
        pipe.fit(X_tr, y_tr)
        pipes[nome] = pipe
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        linhas.append(
            {
                "modelo": nome,
                "CV acurácia": f"{scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}",
                "CV recall": f"{scores['test_recall'].mean():.3f} ± {scores['test_recall'].std():.3f}",
                "CV ROC AUC": f"{scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}",
                "Teste acurácia": f"{accuracy_score(y_te, pred):.3f}",
                "Teste recall": f"{recall_score(y_te, pred):.3f}",
                "Teste ROC AUC": f"{roc_auc_score(y_te, proba):.3f}",
            }
        )

    print(pd.DataFrame(linhas).set_index("modelo").T.to_string())

    melhor = "Random Forest"
    pred = pipes[melhor].predict(X_te)
    print(f"\nRelatório de classificação — {melhor} (conjunto de teste):")
    print(classification_report(y_te, pred, target_names=["Sem doença", "Com doença"]))

    IMAGES_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, display_labels=["Sem doença", "Com doença"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Matriz de confusão — {melhor}")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "matriz_confusao.png", dpi=120)

    rf = pipes[melhor]
    nomes = rf.named_steps["prep"].get_feature_names_out()
    nomes = [n.split("__", 1)[1] for n in nomes]
    imp = pd.Series(rf.named_steps["modelo"].feature_importances_, index=nomes).nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="#4C72B0")
    ax.set_title("Variáveis mais relevantes — Random Forest")
    ax.set_xlabel("Importância")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "importancia_features.png", dpi=120)

    print(f"Gráficos salvos em {IMAGES_DIR}")


if __name__ == "__main__":
    main()
