from __future__ import annotations

"""
CLI para ejecutar EDA en datos crudos y limpios.

Este script:
1. Lee el CSV crudo.
2. Genera un reporte EDA del dataset tal cual.
3. Limpia los datos utilizando `DataCleaner`.
4. Opcionalmente guarda el dataset limpio.
5. Genera un segundo reporte EDA sobre los datos limpios.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from src.data_proc.data_cleaner import CleanConfig, DataCleaner
from src.eda.eda_reporter import EDAReport


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_eda(
    df: pd.DataFrame,
    *,
    title: str,
    report_path: Path,
    summary_path: Path | None,
    fig_dir: Path,
    target: str | None,
    id_cols: List[str],
    datetime_cols: List[str],
    cat_cols: List[str],
    max_plots: int | None,
    drop_label_cols: bool,
    hide_summary_in_html: bool,
) -> Path:
    drop_suffixes = ("_label",) if drop_label_cols else tuple()

    reporter = EDAReport(
        target=target,
        id_cols=id_cols,
        datetime_cols=datetime_cols,
        cat_cols=cat_cols,
        report_title=title,
        fig_dir=str(fig_dir),
        max_plots=max_plots,
        drop_suffixes=drop_suffixes,
        show_exec_summary_in_html=not hide_summary_in_html,
    )

    reporter.fit(df)

    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(reporter.build_executive_summary_text(), encoding="utf-8")

    out_path = Path(reporter.generate_html(str(report_path)))
    return out_path


def _build_clean_config(args: argparse.Namespace) -> CleanConfig:
    return CleanConfig(
        drop_na_thresh=args.drop_na_thresh,
        outlier_z=None if args.outlier_z is None else float(args.outlier_z),
        numeric_like_ratio=args.numeric_like_ratio,
        extra_missing_tokens=_parse_list(args.extra_missing_tokens),
        coerce_float=not args.no_coerce_float,
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera reportes EDA antes y después de limpiar el dataset.")
    parser.add_argument("--raw", required=True, help="Ruta al CSV crudo.")
    parser.add_argument("--clean-out", default="", help="Ruta donde guardar el dataset limpio (opcional).")
    parser.add_argument("--raw-report", default="reports/eda_raw.html", help="Salida HTML para el EDA crudo.")
    parser.add_argument("--clean-report", default="reports/eda_clean.html", help="Salida HTML para el EDA limpio.")
    parser.add_argument("--raw-summary", default="", help="Ruta del resumen ejecutivo (TXT) del EDA crudo.")
    parser.add_argument("--clean-summary", default="", help="Ruta del resumen ejecutivo (TXT) del EDA limpio.")
    parser.add_argument("--raw-title", default="", help="Título para el reporte crudo.")
    parser.add_argument("--clean-title", default="", help="Título para el reporte limpio.")
    parser.add_argument("--raw-fig-dir", default="reports/figures/raw", help="Carpeta para figuras del EDA crudo.")
    parser.add_argument("--clean-fig-dir", default="reports/figures/clean", help="Carpeta para figuras del EDA limpio.")

    parser.add_argument("--target", default="", help="Nombre de la columna objetivo.")
    parser.add_argument("--id-cols", default="", help="Columnas ID, separadas por coma.")
    parser.add_argument("--datetime-cols", default="", help="Columnas de fecha, separadas por coma.")
    parser.add_argument("--cat-cols", default="", help="Columnas categóricas forzadas, separadas por coma.")
    parser.add_argument("--max-plots", type=int, default=None, help="Máximo de gráficas por sección.")
    parser.add_argument("--drop-label-cols", action="store_true", help="Ignora columnas *_label en el EDA.")
    parser.add_argument("--hide-summary-in-html", action="store_true", help="Oculta el resumen en el HTML.")

    parser.add_argument("--drop-na-thresh", type=float, default=0.95, help="Umbral para eliminar columnas con NA.")
    parser.add_argument("--outlier-z", type=float, default=4.0, help="Z score para enmascarar outliers (None para omitir).")
    parser.add_argument(
        "--numeric-like-ratio",
        type=float,
        default=0.8,
        help="Proporción mínima de valores numéricos para convertir una columna string a float.",
    )
    parser.add_argument("--extra-missing-tokens", default="", help="Tokens adicionales que se tratarán como NA.")
    parser.add_argument("--no-coerce-float", action="store_true", help="Desactiva la conversión automática a float.")

    args = parser.parse_args(argv)

    raw_path = Path(args.raw)
    if not raw_path.exists():
        print(f"ERROR: no existe el archivo crudo: {raw_path}", file=sys.stderr)
        return 1

    clean_out_path = Path(args.clean_out) if args.clean_out else None
    raw_report_path = Path(args.raw_report)
    clean_report_path = Path(args.clean_report)
    raw_summary_path = Path(args.raw_summary) if args.raw_summary else None
    clean_summary_path = Path(args.clean_summary) if args.clean_summary else None

    try:
        df_raw = pd.read_csv(raw_path, low_memory=False)
    except Exception as exc:
        print(f"ERROR: no se pudo leer el CSV crudo: {exc}", file=sys.stderr)
        return 1

    target = args.target or None
    id_cols = _parse_list(args.id_cols)
    datetime_cols = _parse_list(args.datetime_cols)
    cat_cols = _parse_list(args.cat_cols)

    raw_title = args.raw_title or f"EDA Crudo — {raw_path.stem}"
    clean_title = args.clean_title or f"EDA Limpio — {raw_path.stem}"

    print(f"[1/4] Generando EDA crudo: {raw_report_path}")
    _run_eda(
        df_raw.copy(),
        title=raw_title,
        report_path=raw_report_path,
        summary_path=raw_summary_path,
        fig_dir=Path(args.raw_fig_dir),
        target=target,
        id_cols=id_cols,
        datetime_cols=datetime_cols,
        cat_cols=cat_cols,
        max_plots=args.max_plots,
        drop_label_cols=args.drop_label_cols,
        hide_summary_in_html=args.hide_summary_in_html,
    )

    print("[2/4] Configurando limpieza")
    clean_config = _build_clean_config(args)
    cleaner = DataCleaner(clean_config)

    print("[3/4] Ejecutando limpieza del dataset")
    df_clean = cleaner.clean(df_raw)
    if clean_out_path:
        cleaner.save(df_clean, clean_out_path)
        print(f"Dataset limpio guardado en: {clean_out_path}")

    print(f"[4/4] Generando EDA limpio: {clean_report_path}")
    _run_eda(
        df_clean,
        title=clean_title,
        report_path=clean_report_path,
        summary_path=clean_summary_path,
        fig_dir=Path(args.clean_fig_dir),
        target=target,
        id_cols=id_cols,
        datetime_cols=datetime_cols,
        cat_cols=cat_cols,
        max_plots=args.max_plots,
        drop_label_cols=args.drop_label_cols,
        hide_summary_in_html=args.hide_summary_in_html,
    )

    print("OK: flujo de EDA crudo + limpio finalizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

