from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict
from pathlib import Path

import numpy as np
import pandas as pd

# Gráficas opcionales (se usan sólo en helpers)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# =========================
# Configuración y utilidades
# =========================

@dataclass(frozen=True)
class ProjectConfig:
    target: str = "CARAVAN"
    cat_cols: Tuple[str, ...] = ("MOSTYPE", "MOSHOOFD", "MKOOPKLA")
    extra_exclusions: Tuple[str, ...] = ("MBERBOER",)  # variables a excluir siempre


# =========================
# 1) Detección de outliers
# =========================

@dataclass
class OutlierResults:
    bounds: pd.DataFrame
    outliers_long: pd.DataFrame
    counts: pd.Series


class OutlierDetectorIQR:
    """
    Detecta outliers por IQR y calcula extremos relativos (min/Q1 y max/Q3).
    """
    def __init__(self, k: float = 1.5, exclude: Optional[Iterable[str]] = None):
        self.k = k
        self.exclude = set(exclude or [])

    def detect(
        self,
        df: pd.DataFrame,
        cols: Optional[Iterable[str]] = None
    ) -> OutlierResults:
        num_cols = cols or df.select_dtypes(include="number").columns.difference(self.exclude).tolist()
        if not num_cols:
            return OutlierResults(
                bounds=pd.DataFrame(columns=["Q1","Q3","IQR","lower","upper","extrem_min","extrem_max"]),
                outliers_long=pd.DataFrame(columns=["index","col","value","lower","upper","extrem_min","extrem_max"]),
                counts=pd.Series(dtype="int64")
            )

        q1 = df[num_cols].quantile(0.25)
        q3 = df[num_cols].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - self.k * iqr
        upper = q3 + self.k * iqr

        with np.errstate(divide="ignore", invalid="ignore"):
            extrem_max = (df[num_cols].max() / q3.replace(0, np.nan)).round(2)
            extrem_min = (df[num_cols].min() / q1.replace(0, np.nan)).round(2)

        bounds = pd.DataFrame(
            {"Q1": q1, "Q3": q3, "IQR": iqr, "lower": lower, "upper": upper,
             "extrem_min": extrem_min, "extrem_max": extrem_max}
        )

        parts = []
        for c in num_cols:
            s = df[c]
            m = (s < lower[c]) | (s > upper[c])
            if m.any():
                parts.append(pd.DataFrame({
                    "index": s.index[m],
                    "col": c,
                    "value": s[m].values,
                    "lower": lower[c],
                    "upper": upper[c],
                    "extrem_min": extrem_min[c],
                    "extrem_max": extrem_max[c],
                }))

        outliers_long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["index","col","value","lower","upper","extrem_min","extrem_max"]
        )
        counts = outliers_long["col"].value_counts().rename("n_outliers").sort_values(ascending=False)
        return OutlierResults(bounds=bounds, outliers_long=outliers_long, counts=counts)


# ==================================================
# 2) Política para decidir variables a no depurar
#     (mantener outliers según percentil de extrem_max)
# ==================================================

class OutlierPolicy:
    """
    Decide qué variables NO depurar (mantener outliers) según p-quantil de extrem_max.
    """
    def __init__(self, quantile: float = 0.15):
        self.quantile = quantile

    def vars_to_keep(self, bounds: pd.DataFrame) -> List[str]:
        b = bounds.copy()
        b = b[b["extrem_max"].notna()]
        if b.empty:
            return []
        p = float(b["extrem_max"].quantile(self.quantile))
        keep = b.loc[b["extrem_max"] <= p].index.tolist()
        return keep


# ===========================================
# 3) Análisis por variable (sin/ con outliers)
# ===========================================

@dataclass
class VariableDecision:
    variable: str
    considerar: str   # "SI"/"NO"
    cambia_rango: str # "SI"/"NO"
    long_media: float
    dif_mean: float
    tabla_ct: pd.DataFrame


class VariableAnalyzer:
    """
    Replica la lógica de 'info_variable': calcula stats por clase (sin outliers),
    decide considerar/cambiar rango, y puede graficar.
    """
    def __init__(self, target: str):
        self.target = target

    @staticmethod
    def remove_outliers_of_vars(df: pd.DataFrame, vars_: List[str], outliers_long: pd.DataFrame) -> pd.DataFrame:
        if outliers_long is None or outliers_long.empty or not vars_:
            return df.copy()
        idx = outliers_long.loc[outliers_long['col'].isin(vars_), 'index'].unique()
        return df.drop(index=idx, errors="ignore").reset_index(drop=True)

    def analyze(
        self,
        df: pd.DataFrame,
        var: str,
        outliers_long: pd.DataFrame,
        bounds: pd.DataFrame,
        plot: bool = False
    ) -> VariableDecision:
        df_wo = self.remove_outliers_of_vars(df, [var], outliers_long)
        tabla_ct = (
            df_wo.groupby(self.target)[var]
                 .agg(count="count", mean="mean", median="median", std="std", min='min', max='max')
                 .round(2)
                 .reindex([0, 1])
        )

        # métricas de decisión
        pct_1 = round(100 * tabla_ct.loc[1, 'count'] / (tabla_ct.loc[0, 'count'] + tabla_ct.loc[1, 'count']), 2) if tabla_ct.loc[1, "count"] > 0 else 0.0
        dif_mean = round(abs(tabla_ct.loc[0, 'mean'] - tabla_ct.loc[1, 'mean']) / (tabla_ct.loc[0, 'mean'] if tabla_ct.loc[0, 'mean'] != 0 else 1e-9), 2)
        long_media = round(abs(tabla_ct.loc[0, 'max'] - tabla_ct.loc[0, 'min']), 2) > 0
        # relación de rangos
        denom = abs(tabla_ct.loc[1, 'max'] - tabla_ct.loc[1, 'min'])
        denom = denom if denom != 0 else 1e-9
        dif_media = round(abs(tabla_ct.loc[0, 'max'] - tabla_ct.loc[0, 'min']) / denom, 2)

        cambia_rango = "NO" if (0.9 < dif_media < 1.10 and long_media is True) else "SI"
        considerar = "NO" if (dif_mean < 0.10) else "SI"

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            df_wo[var].plot(kind="hist", bins=30, ax=axes[0], title=f"Hist {var} (sin outliers)")
            sns.boxplot(x=self.target, y=var, data=df_wo, ax=axes[1])
            plt.tight_layout(); plt.show()

        return VariableDecision(
            variable=var,
            considerar=considerar,
            cambia_rango=cambia_rango,
            long_media=float(long_media),
            dif_mean=float(dif_mean),
            tabla_ct=tabla_ct
        )


# ===========================================
# 4) Screening de variables + diagnóstico
# ===========================================

class FeatureScreener:
    """
    Orquesta el análisis variable por variable y devuelve:
    - df_diagnostico (variable, considerar, cambia_rango)
    - listas de exclusión basadas en política y diagnóstico
    """
    def __init__(self, config: ProjectConfig):
        self.cfg = config
        self.analyzer = VariableAnalyzer(target=self.cfg.target)

    def run(
        self,
        df_clean: pd.DataFrame,
        outliers: OutlierResults,
        keep_outlier_vars: List[str]
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        # columnas a excluir de entrada
        exclude_cols = list(self.cfg.cat_cols) + keep_outlier_vars + list(self.cfg.extra_exclusions)

        df_dep = df_clean.drop(columns=exclude_cols, errors="ignore")
        resultados = []
        for col in df_dep.columns:
            if col == self.cfg.target:
                continue
            dec = self.analyzer.analyze(df_clean, col, outliers.outliers_long, outliers.bounds, plot=False)
            resultados.append(pd.Series([dec.variable, dec.considerar, dec.cambia_rango]))

        df_diag = pd.DataFrame(resultados)
        # Filtrado: quedarnos con variables a NO considerar o NO cambia rango
        df_filt = df_diag[(df_diag.iloc[:, 1] == "NO") | (df_diag.iloc[:, 2] == "NO")]
        var_no_consider = df_filt.iloc[:, 0].tolist()

        # Exclusiones finales (Fase 1)
        exclude_final = keep_outlier_vars + list(self.cfg.extra_exclusions) + var_no_consider
        return df_diag, var_no_consider, exclude_final


# ===========================================
# 5) Pruning por correlación
# ===========================================

class CorrelationPruner:
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def corr_pairs(self, df: pd.DataFrame) -> pd.Series:
        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_upper = corr.where(mask)
        pairs = (
            corr_upper.stack()
                      .rename("r")
                      .pipe(lambda s: s[abs(s) >= self.threshold])
                      .sort_values(key=np.abs, ascending=False)
        )
        return pairs

    def prune(self, df: pd.DataFrame, extra_drop: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str], pd.Series]:
        pairs = self.corr_pairs(df)
        to_drop = list(extra_drop or [])
        # Estrategia simple: si hay lista manual, la aplicamos; si no, no quitamos por pares automáticamente.
        df2 = df.drop(columns=to_drop, errors="ignore")
        return df2, to_drop, pairs


# ===========================================
# 6) PCA + Split
# ===========================================

@dataclass
class PCAResult:
    X_pca: pd.DataFrame
    pca: PCA
    scaler: StandardScaler
    explained_var_pct: np.ndarray
    feature_names: list 
    

class PCAPipeline:
    def __init__(self, n_components: int = 15):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)

    def fit_transform(self, X: pd.DataFrame) -> PCAResult:
        X_num = X.select_dtypes(include="number").copy()
        feature_names = X_num.columns.tolist()

        X_scaled = self.scaler.fit_transform(X_num)
        X_pca = self.pca.fit_transform(X_scaled)
        var_pct = np.round(self.pca.explained_variance_ratio_ * 100, 2)
        df_pca = pd.DataFrame(
            X_pca, 
            columns=[f"PC{i+1}" for i in range(self.pca.n_components_)], 
            index=X.index
        )
        return PCAResult(
            X_pca=df_pca, 
            pca=self.pca, 
            scaler=self.scaler, 
            explained_var_pct=var_pct,
            feature_names=feature_names 
        )

    @staticmethod
    def train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ===========================================
# 7) Orquestador (end-to-end “ligero”)
# ===========================================

class AnalysisOrchestrator:
    """
    Ejecuta: outliers -> política -> screening -> pruning -> PCA -> split.
    Deja ganchos para que el notebook decida graficar o inspeccionar resultados.
    """
    def __init__(self, config: ProjectConfig, outlier_k: float = 1.5, policy_quantile: float = 0.15, corr_threshold: float = 0.3, pca_components: int = 15):
        self.cfg = config
        self.detector = OutlierDetectorIQR(k=outlier_k, exclude=[self.cfg.target])
        self.policy = OutlierPolicy(quantile=policy_quantile)
        self.screener = FeatureScreener(self.cfg)
        self.pruner = CorrelationPruner(threshold=corr_threshold)
        self.pca_pipe = PCAPipeline(n_components=pca_components)


    def run(self, df_clean: pd.DataFrame, manual_drop: Optional[List[str]] = None, verbose: bool = True) -> Dict[str, object]:
        # 1) Outliers
        df_sin_cat = df_clean.drop(columns=list(self.cfg.cat_cols), errors="ignore")
        out_res = self.detector.detect(df_sin_cat)

        # 2) Política de mantener outliers
        keep_vars = self.policy.vars_to_keep(out_res.bounds)

        # 3) Screening variables y exclusiones iniciales
        df_diag, var_no_consider, exclude_final = self.screener.run(df_clean, out_res, keep_vars)

        # 4) Construcción df_mod y pruning por correlación
        df_mod = df_clean.drop(columns=exclude_final, errors="ignore")
        if manual_drop:
            df_mod = df_mod.drop(columns=manual_drop, errors="ignore")
        pairs = self.pruner.corr_pairs(df_mod)

        # 5) Separar X, y y aplicar PCA
        X = df_mod.drop(columns=[self.cfg.target], errors="ignore")
        y = df_mod[self.cfg.target].copy()
        pca_res = self.pca_pipe.fit_transform(X)

        if verbose:
            vars_pca = pca_res.feature_names
            vars_pca = pca_res.feature_names
            x_shape = X.select_dtypes(include="number").shape
            y_shape = y.shape
            n_components = pca_res.pca.n_components_
            pca_cols = pca_res.X_pca.shape[1]
            var_pct = pca_res.explained_var_pct
            var_pct_acum = float(var_pct.sum())

            print("Dimensiones de X (antes de PCA):", x_shape)
            print("Variables usadas en PCA ({}): {}".format(len(vars_pca), vars_pca))
            print("CARAVAN_in_PCA:", ("CARAVAN" in vars_pca))  # Imprime False si NO está
            print("Componentes PCA:", n_components)
            print("Columnas en X_pca:", pca_cols)
            #print("Varianza explicada por componente (%):", pca_res.explained_var_pct.tolist())
            print("Varianza explicada acumulada (%):", float(pca_res.explained_var_pct.sum()))



        # 6) Split
        X_train, X_test, y_train, y_test = self.pca_pipe.train_test(pca_res.X_pca, y)

        return {
            "outliers": out_res,
            "keep_outlier_vars": keep_vars,
            "diagnostico": df_diag,
            "exclude_final": exclude_final + list(manual_drop or []),
            "corr_pairs": pairs,
            "pca_result": pca_res,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "df_mod": df_mod
         }


# =========================
# Helpers de visualización
# =========================

def plot_corr_with_target(df: pd.DataFrame, target: str, thr: float = 0.1):
    corr_target = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    corr_filtrada = corr_target[(abs(corr_target) > thr)]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=corr_filtrada.values, y=corr_filtrada.index, hue=corr_filtrada.index, legend=False)
    plt.title(f"Correlación con el objetivo: {target}")
    plt.xlabel("Coef. de correlación"); plt.ylabel("Variable")
    plt.tight_layout(); plt.show()
    return corr_filtrada
