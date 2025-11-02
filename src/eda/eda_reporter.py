# src/eda/eda_reporter.py
"""
EDA Reporter (OOP)
------------------
Genera un reporte HTML con tablas y gráficos a partir de un DataFrame y permite exportar
un RESUMEN EJECUTIVO en texto plano (sin Markdown).

Uso básico:
-----------
from src.eda.eda_reporter import EDAReport

r = EDAReport(
    target="CARAVAN",  # o "CARAVAN_BIN"
    id_cols=[],
    datetime_cols=[],
    cat_cols=[],
    report_title="EDA — CARAVAN",
    drop_suffixes=("_label",),  # ignora columnas *_label (útil en tu dataset)
    max_plots=None              # None = mostrar TODO; o pasa un entero para limitar
)
r.fit(df)
txt = r.build_executive_summary_text()
r.generate_html("reports/eda_report.html")
"""

import os
import io
import base64
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_SM = True
except Exception:
    HAS_SM = False


class EDAReport:
    def __init__(
        self,
        target=None,
        id_cols=None,
        datetime_cols=None,
        cat_cols=None,
        report_title="EDA Report",
        fig_dir="reports/figures",
        max_plots=None,              # None = mostrar TODO; entero para limitar
        drop_suffixes=("_label",),   # útil para ignorar columnas descriptivas *_label
        show_exec_summary_in_html=True,  # muestra el resumen ejecutivo en el HTML
        missing_flag=0.30,               # umbral %NA alto
        vif_flag=10.0,                   # umbral VIF alto
        skew_flag=1.0,                   # umbral |skew| alto
        pct_zero_flag=0.80               # umbral %ceros alto
    ):
        self.target = target
        self.id_cols = id_cols or []
        self.datetime_cols = datetime_cols or []
        self.cat_cols_user = cat_cols or []
        self.report_title = report_title
        self.fig_dir = fig_dir
        self.max_plots = max_plots
        self.drop_suffixes = tuple(drop_suffixes) if drop_suffixes else tuple()
        self.show_exec_summary_in_html = show_exec_summary_in_html

        # thresholds para el resumen ejecutivo
        self.missing_flag = missing_flag
        self.vif_flag = vif_flag
        self.skew_flag = skew_flag
        self.pct_zero_flag = pct_zero_flag

        os.makedirs(self.fig_dir, exist_ok=True)

        self.df = None
        self.num_cols = []
        self.cat_cols = []
        self.summary_numeric_ = None
        self.missing_ = None
        self.corr_ = None
        self.vif_ = None
        self.findings_ = []

    # -------------------- helpers de tipo/filtrado --------------------
    def _drop_by_suffix(self, df: pd.DataFrame):
        if not self.drop_suffixes:
            return df
        keep = [c for c in df.columns if not any(c.endswith(suf) for suf in self.drop_suffixes)]
        return df[keep].copy()

    def _infer_types(self, df: pd.DataFrame):
        user_cats = set(self.cat_cols_user)
        inferred_cats = {
            c for c in df.columns
            if (df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c]))
        }
        self.cat_cols = list(user_cats.union(inferred_cats))
        self.cat_cols = [c for c in self.cat_cols if c not in (self.id_cols + self.datetime_cols)]

        self.num_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in (self.id_cols + self.datetime_cols + self.cat_cols)
        ]

    # -------------------- API pública --------------------
    def fit(self, df: pd.DataFrame):
        # filtra *_label u otros sufijos si se pidió
        df = self._drop_by_suffix(df)
        self.df = df.copy()

        # parsea fechas
        for c in self.datetime_cols:
            self.df[c] = pd.to_datetime(self.df[c], errors="coerce")

        # infiere tipos
        self._infer_types(self.df)

        # faltantes
        self.missing_ = self.df.isna().mean().sort_values(ascending=False).to_frame("%NA")

        # resumen numérico
        if self.num_cols:
            desc = self.df[self.num_cols].describe().T
            desc["skew"] = self.df[self.num_cols].skew()
            desc["kurt"] = self.df[self.num_cols].kurt()
            desc["pct_zeros"] = (self.df[self.num_cols] == 0).mean()
            self.summary_numeric_ = desc

        # correlación entre FEATURES numéricas (no con el target) usando solo columnas realmente numéricas
        if len(self.num_cols) >= 2:
            num_df = self.df[self.num_cols].select_dtypes(include=[np.number])
            if num_df.shape[1] >= 2:
                self.corr_ = num_df.corr(method="spearman")

        # VIF (robusto)
        if HAS_SM and len(self.num_cols) >= 2:
            X = self.df[self.num_cols].dropna()
            if len(X) > 0:
                self.vif_ = self._compute_vif_safe(X)

        return self

    # -------------------- VIF robusto --------------------
    def _compute_vif_safe(self, X: pd.DataFrame, drop_high_corr=True, corr_threshold=0.999):
        """
        Calcula VIF de forma robusta:
        - Quita constantes o columnas con varianza ~0
        - Quita duplicadas exactas
        - (opcional) Quita una de cada par con |corr|>=corr_threshold (evita R^2≈1)
        Devuelve un DataFrame con columnas: variable, VIF
        """
        if not HAS_SM:
            return pd.DataFrame(columns=["variable", "VIF"])

        Z = X.select_dtypes(include=[np.number]).copy()
        Z = Z.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

        # 1) constantes/varianza 0
        var0 = [c for c in Z.columns if Z[c].std(ddof=0) == 0]
        if var0:
            Z = Z.drop(columns=var0)

        # 2) duplicadas exactas
        if Z.shape[1] > 1:
            dup_mask = Z.T.duplicated(keep="first")
            Z = Z.loc[:, ~dup_mask]

        # 3) (opcional) poda por correlación extrema
        if drop_high_corr and Z.shape[1] > 1:
            corr = Z.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] >= corr_threshold)]
            if to_drop:
                Z = Z.drop(columns=to_drop)

        if Z.shape[1] <= 1:
            return pd.DataFrame(columns=["variable", "VIF"])

        Zc = sm.add_constant(Z)
        rows = []
        for i, col in enumerate(Zc.columns):
            if col == "const":
                continue
            try:
                vif_val = variance_inflation_factor(Zc.values, i)
                if np.isfinite(vif_val):
                    rows.append({"variable": col, "VIF": float(vif_val)})
            except Exception:
                # si statsmodels falla por algún motivo, lo saltamos
                continue

        return pd.DataFrame(rows).sort_values("VIF", ascending=False)

    # -------------------- Executive summary --------------------
    def build_executive_summary_text(self, max_examples=5):
        lines = []
        # 1) Contexto general
        n_rows = len(self.df)
        n_cols = len(self.df.columns)
        lines.append(f"RESUMEN EJECUTIVO — {self.report_title}")
        lines.append(f"Registros: {n_rows:,}  |  Columnas: {n_cols:,}")
        lines.append(f"Numéricas: {len(self.num_cols)}  |  Categóricas: {len(self.cat_cols)}  |  Fechas: {len(self.datetime_cols)}")
        lines.append(f"Target: {self.target if self.target else '—'}")

        # 2) Calidad de datos
        high_missing = []
        if self.missing_ is not None:
            high_missing = self.missing_["%NA"][self.missing_["%NA"] >= self.missing_flag].index.tolist()
        lines.append(f"Columnas con NA > {int(self.missing_flag*100)}%: {len(high_missing)}" + (f" | Ejemplos: {', '.join(high_missing[:max_examples])}" if high_missing else ""))

        # 3) Sesgo, ceros y VIF
        skewed_vars = []
        high_zero_vars = []
        if self.summary_numeric_ is not None:
            skewed_vars = self.summary_numeric_[self.summary_numeric_["skew"].abs() > self.skew_flag].index.tolist()
            high_zero_vars = self.summary_numeric_[self.summary_numeric_["pct_zeros"] >= self.pct_zero_flag].index.tolist()
        lines.append(f"Variables con |skew| > {self.skew_flag}: {len(skewed_vars)}" + (f" | Ejemplos: {', '.join(skewed_vars[:max_examples])}" if skewed_vars else ""))
        lines.append(f"Variables con %ceros ≥ {int(self.pct_zero_flag*100)}%: {len(high_zero_vars)}" + (f" | Ejemplos: {', '.join(high_zero_vars[:max_examples])}" if high_zero_vars else ""))

        high_vif = []
        if self.vif_ is not None:
            high_vif = self.vif_[self.vif_["VIF"] > self.vif_flag]["variable"].tolist()
        lines.append(f"Variables con VIF > {self.vif_flag}: {len(high_vif)}" + (f" | Ejemplos: {', '.join(high_vif[:max_examples])}" if high_vif else ""))

        # 4) Relación entre features numéricas (correlación fuerte)
        strong_pairs = []
        if self.corr_ is not None:
            corr_mat = self.corr_.copy()
            cols = corr_mat.columns.tolist()
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    r = corr_mat.iloc[i, j]
                    if pd.notna(r) and abs(r) >= 0.7:
                        strong_pairs.append((cols[i], cols[j], float(r)))
        strong_pairs_sorted = sorted(strong_pairs, key=lambda x: abs(x[2]), reverse=True)[:max_examples]
        if strong_pairs_sorted:
            pairs_txt = "; ".join([f"{a} ~ {b} (r={r:.2f})" for a, b, r in strong_pairs_sorted])
            lines.append(f"Pares de features con correlación |r|≥0.70: {len(strong_pairs)} | Ejemplos: {pairs_txt}")
        else:
            lines.append("No se detectaron pares de features con correlación |r|≥0.70.")

        # 5) Relación con el target
        if self.target and self.target in self.df.columns:
            y = self.df[self.target]
            if pd.api.types.is_numeric_dtype(y) and self.num_cols:
                rows = []
                for c in self.num_cols:
                    r, p = stats.spearmanr(self.df[c], y, nan_policy="omit")
                    if pd.notna(r):
                        rows.append((c, float(r), float(p) if pd.notna(p) else np.nan))
                rows_sorted = sorted(rows, key=lambda x: abs(x[1]), reverse=True)[:max_examples]
                if rows_sorted:
                    lines.append("Top relaciones (Spearman) numéricas vs target:")
                    for name, r, p in rows_sorted:
                        lines.append(f"  - {name}: r={r:.3f}, p={p:.1e}")
                else:
                    lines.append("No se detectaron correlaciones numéricas significativas con el target.")
            else:
                # Target categórico: para cada categórica, máxima concentración por categoría
                cat_scores = []
                for c in self.cat_cols:
                    try:
                        ct = pd.crosstab(self.df[c], self.df[self.target], normalize="index")
                        max_conc = float(ct.max(axis=1).mean())  # promedio del máximo por fila
                        cat_scores.append((c, max_conc))
                    except Exception:
                        continue
                cat_sorted = sorted(cat_scores, key=lambda x: x[1], reverse=True)[:max_examples]
                if cat_sorted:
                    lines.append("Top variables categóricas por concentración promedio del target (0–1):")
                    for name, score in cat_sorted:
                        lines.append(f"  - {name}: {score:.3f}")
                else:
                    lines.append("No se detectaron variables categóricas con concentración destacada hacia el target.")
        else:
            lines.append("Sin target definido.")

        # 6) Recomendaciones operativas
        recs = []
        if len(skewed_vars) > 0:
            recs.append("Aplicar transformaciones (log/yeo-johnson) o modelos robustos por asimetría elevada.")
        if len(high_zero_vars) > 0:
            recs.append("Tratar alta proporción de ceros (binarizar flags o modelos con masa en cero).")
        if len(high_missing) > 0:
            recs.append("Imputación avanzada o descarte de columnas con NA elevados (>={}% ).".format(int(self.missing_flag*100)))
        if len(high_vif) > 0:
            recs.append("Reducir multicolinealidad (selección, combinación o regularización).")
        if not recs:
            recs.append("Datos en estado adecuado; avanzar a ingeniería de variables y modelado.")
        lines.append("Recomendaciones:")
        for rtxt in recs[:max_examples]:
            lines.append(f"  - {rtxt}")

        return "\n".join(lines)

    def generate_html(self, output_path="reports/eda_report.html"):
        html = f"""
        <html>
        <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=device-width, initial-scale=1"/>
          {self._style()}
          <title>{self.report_title}</title>
        </head>
        <body>
            {self._header()}
            {self._section_exec_summary() if self.show_exec_summary_in_html else ""}
            {self._section_missing()}
            {self._section_univariate()}
            {self._section_corr_vif()}
            {self._section_target()}
            {self._section_findings()}
        </body>
        </html>
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    # -------------------- plotting --------------------
    def _save_fig_as_b64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    def _plot_hist(self, series, title):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(series.dropna(), bins=30)
        ax.set_title(title)
        return self._save_fig_as_b64(fig)

    def _plot_bar(self, series, title, topn=None):
        vc = series.value_counts()
        if topn is not None and topn > 0:
            vc = vc.head(topn)
        x = list(range(len(vc)))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(x, vc.values)
        ax.set_xticks(x)
        ax.set_xticklabels(vc.index.astype(str), rotation=45, ha="right")
        ax.set_title(title)
        return self._save_fig_as_b64(fig)

    def _plot_scatter(self, x, y, title):
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.scatter(x, y, alpha=0.5, s=10)
        ax.set_title(title)
        return self._save_fig_as_b64(fig)

    # -------------------- HTML helpers --------------------
    def _style(self):
        return """
        <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;background:#0b1020;color:#e8eaf3;margin:0;padding:2rem;}
        h1,h2,h3{color:#fff;margin:0.75rem 0;}
        .card{background:#121735;border:1px solid #1f2750;border-radius:16px;padding:16px;margin:14px 0;box-shadow:0 6px 24px rgba(0,0,0,.35)}
        .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;}
        .muted{color:#b7bdd9;font-size:0.95rem}
        .table{width:100%;border-collapse:collapse;border-spacing:0;font-size:0.9rem}
        .table th,.table td{border-bottom:1px dashed #2a356e;padding:8px 6px;text-align:left}
        .table th{position:sticky;top:0;background:#121735;z-index:1}
        .kpi{display:flex;gap:12px;flex-wrap:wrap}
        .pill{background:#0f1a4a;border:1px solid #2941a8;color:#c9d3ff;border-radius:999px;padding:6px 10px;font-size:.85rem}
        img{max-width:100%;height:auto;border-radius:12px;border:1px solid #243071}
        .caption{color:#9aa4d6;font-size:0.85rem;margin-top:6px}
        code{background:#0f1a4a;padding:2px 6px;border-radius:8px}
        pre.execsum{white-space:pre-wrap;background:#0f1a4a;border:1px solid #2941a8;border-radius:12px;padding:12px;color:#e8eaf3}

        /* NUEVO: contenedor con scroll para tablas grandes */
        .scrollbox{overflow:auto;max-width:100%;max-height:60vh;border:1px solid #1f2750;border-radius:12px;}
        .scrollbox table{width:max-content;min-width:100%;}
        </style>
        """

    def _header(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""
        <h1>{self.report_title}</h1>
        <p class='muted'>Generado: {now}</p>
        <div class='kpi'>
            <span class='pill'>Filas: {len(self.df):,}</span>
            <span class='pill'>Columnas: {len(self.df.columns):,}</span>
            <span class='pill'>Numéricas: {len(self.num_cols)}</span>
            <span class='pill'>Categóricas: {len(self.cat_cols)}</span>
            <span class='pill'>Fechas: {len(self.datetime_cols)}</span>
            <span class='pill'>Target: {self.target if self.target else '—'}</span>
        </div>
        """

    def _tbl_html(self, df, max_rows=30):
        return df.head(max_rows).to_html(border=0, classes="table table-sm", index=True)

    def _section_exec_summary(self):
        txt = self.build_executive_summary_text()
        return f"""
        <div class='card'>
          <h2>Resumen ejecutivo</h2>
          <pre class='execsum'>{txt}</pre>
        </div>
        """

    def _section_missing(self):
        return f"""
        <div class='card'>
          <h2>Faltantes</h2>
          <p class='muted'>Porcentaje de NA por columna (top 30).</p>
          <div class='scrollbox'>
            {self._tbl_html(self.missing_)}
          </div>
        </div>
        """

    def _iter_limit(self, cols):
        if self.max_plots is None or self.max_plots == 0:
            return cols
        return cols[: self.max_plots]

    def _section_univariate(self):
        blocks = []
        if self.summary_numeric_ is not None:
            blocks.append(f"""
            <div class='card'>
              <h2>Univariado — Numéricas</h2>
              <p class='muted'>Resumen estadístico (incluye skew, kurtosis, % ceros).</p>
              <div class='scrollbox'>
                {self._tbl_html(self.summary_numeric_.round(3))}
              </div>
            </div>
            """)
            imgs = []
            for c in self._iter_limit(self.num_cols):
                imgs.append(f"<div><img src='{self._plot_hist(self.df[c], f'Histograma — {c}')}'/><div class='caption'>{c}</div></div>")
            if imgs:
                blocks.append("<div class='grid'>" + "".join(imgs) + "</div>")

        if self.cat_cols:
            blocks.append(f"""
            <div class='card'>
              <h2>Univariado — Categóricas</h2>
              <p class='muted'>Frecuencias por categoría.</p>
            </div>
            """)
            imgs = []
            for c in self._iter_limit(self.cat_cols):
                imgs.append(f"<div><img src='{self._plot_bar(self.df[c], f'Frecuencias — {c}', topn=None)}'/><div class='caption'>{c}</div></div>")
            if imgs:
                blocks.append("<div class='grid'>" + "".join(imgs) + "</div>")

        return "".join(blocks)

    def _section_corr_vif(self):
        blocks = []
        if self.corr_ is not None:
            blocks.append(f"""
            <div class='card'>
            <h2>Correlaciones (Spearman) entre features numéricas</h2>
            <p class='muted'>Matriz mostrada como tabla (valores redondeados).</p>
            <div class='scrollbox'>
                {self._tbl_html(self.corr_.round(3))}
            </div>
            </div>
            """)

        if self.vif_ is not None and len(self.vif_) > 0:
            blocks.append(f"""
            <div class='card'>
            <h2>VIF</h2>
            <p class='muted'>Factores de inflación de varianza para detectar multicolinealidad.</p>
            <div class='scrollbox'>
                {self._tbl_html(self.vif_.round(3))}
            </div>
            </div>
            """)

        return "".join(blocks) if blocks else ""

    def _section_target(self):
        if not self.target or self.target not in self.df.columns:
            return ""
        y = self.df[self.target]
        blocks = [f"<div class='card'><h2>Con el Target ({self.target})</h2>"]

        if pd.api.types.is_numeric_dtype(y):
            rows = []
            for c in self.num_cols:
                r, p = stats.spearmanr(self.df[c], y, nan_policy="omit")
                rows.append({"feature": c, "spearman_r": r, "p_value": p})
            tbl = pd.DataFrame(rows).sort_values("spearman_r", ascending=False)
            blocks.append("<p class='muted'>Correlaciones (Spearman) entre variables numéricas y el target.</p><div class='scrollbox'>" + self._tbl_html(tbl.round(4)) + "</div>")

            imgs = []
            for c in self._iter_limit(list(tbl["feature"])):
                imgs.append(f"<div><img src='{self._plot_scatter(self.df[c], y, f'{c} vs {self.target}')}'/><div class='caption'>{c}</div></div>")
            if imgs:
                blocks.append("<div class='grid'>" + "".join(imgs) + "</div>")

        else:
            blocks.append("<p class='muted'>Distribución del target por cada variable categórica (barras apiladas).</p>")
            imgs = []

            target_order = (
                self.df[self.target]
                .value_counts()
                .index.tolist()
            )

            for c in self._iter_limit(self.cat_cols):
                vc = pd.crosstab(self.df[c], self.df[self.target], normalize="index")
                vc = vc.reindex(columns=target_order).fillna(0.0)
                fig, ax = plt.subplots(figsize=(6, 3))
                vc.plot(kind="bar", stacked=True, ax=ax, legend=True)
                ax.set_title(f"{c} vs {self.target}")
                ax.legend(title=str(self.target), bbox_to_anchor=(1.02, 1), loc="upper left")
                imgs.append(f"<div><img src='{self._save_fig_as_b64(fig)}'/><div class='caption'>{c}</div></div>")
            if imgs:
                blocks.append("<div class='grid'>" + "".join(imgs) + "</div>")

        blocks.append("</div>")
        return "".join(blocks)

    def _section_findings(self):
        if not self.findings_:
            recs = []
            if self.summary_numeric_ is not None:
                skewed = self.summary_numeric_["skew"].abs().sort_values(ascending=False)
                if (skewed > self.skew_flag).any():
                    recs.append("Variables fuertemente sesgadas (|skew| alto). Considera log/yeo-johnson o modelos robustos.")
            if self.missing_ is not None and (self.missing_["%NA"] > self.missing_flag).any():
                recs.append("Columnas con NA elevados: evaluar descartar o imputar por grupo/algoritmo.")
            if self.vif_ is not None and len(self.vif_) and (self.vif_["VIF"] > self.vif_flag).any():
                recs.append("Multicolinealidad alta (VIF). Considera eliminar/combinar variables.")

            if not recs:
                recs.append("Datos en buen estado general; procede a ingeniería de variables y modelado.")
            self.findings_ = recs

        items = "".join([f"<li>{x}</li>" for x in self.findings_])
        return f"""
        <div class='card'>
          <h2>Hallazgos y próximas acciones</h2>
          <ul>{items}</ul>
        </div>
        """
