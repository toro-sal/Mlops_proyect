# src/run_eda.py
import os
import sys
import argparse
import pandas as pd

from src.eda.eda_reporter import EDAReport


def _parse_list(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _binarize_target_inplace(df: pd.DataFrame, target: str) -> str:
    new_col = f"{target}_BIN"
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no existe en el DataFrame.")
    ser = pd.to_numeric(df[target], errors="coerce").fillna(0)
    df[new_col] = (ser > 0).astype(int)
    return new_col


def main():
    p = argparse.ArgumentParser(description="Genera un reporte EDA HTML y un resumen ejecutivo en texto plano.")
    p.add_argument("--csv", required=True, help="Ruta al CSV de entrada.")
    p.add_argument("--out", required=True, help="Ruta del HTML de salida, e.g. reports/eda.html")
    p.add_argument("--target", required=False, help="Nombre del target (columna objetivo).")
    p.add_argument("--binarize-target", action="store_true", help="Si se pasa, crea {target}_BIN y lo usa.")
    p.add_argument("--drop-label-cols", action="store_true", help="Ignora columnas *_label")
    p.add_argument("--id-cols", default="", help="Columnas ID, coma-separadas.")
    p.add_argument("--datetime-cols", default="", help="Columnas de fecha, coma-separadas.")
    p.add_argument("--cat-cols", default="", help="Forzar columnas categóricas, coma-separadas.")
    p.add_argument("--max-plots", type=int, default=None, help="Máximo de gráficos por bloque (omitir para 'todas').")
    p.add_argument("--report-title", default="EDA Report", help="Título mostrado en el HTML.")
    p.add_argument("--summary-txt", default="", help="Ruta de salida para el resumen ejecutivo en TXT (opcional).")
    p.add_argument("--hide-summary-in-html", action="store_true", help="No mostrar el resumen en el HTML.")

    args = p.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"ERROR: no existe el archivo CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"ERROR: no se pudo leer el CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("ERROR: el DataFrame está vacío.", file=sys.stderr)
        sys.exit(1)

    id_cols = _parse_list(args.id_cols)
    datetime_cols = _parse_list(args.datetime_cols)
    cat_cols = _parse_list(args.cat_cols)

    drop_suffixes = ("_label",) if args.drop_label_cols else tuple()

    target = args.target
    if target and args.binarize_target:
        try:
            target = _binarize_target_inplace(df, target)
        except Exception as e:
            print(f"ERROR al binarizar target: {e}", file=sys.stderr)
            sys.exit(1)

    reporter = EDAReport(
        target=target,
        id_cols=id_cols,
        datetime_cols=datetime_cols,
        cat_cols=cat_cols,
        report_title=args.report_title,
        fig_dir="reports/figures",
        max_plots=args.max_plots,
        drop_suffixes=drop_suffixes,
        show_exec_summary_in_html=not args.hide_summary_in_html
    )

    try:
        reporter.fit(df)

        # Resumen ejecutivo en TXT (opcional)
        if args.summary_txt:
            os.makedirs(os.path.dirname(args.summary_txt), exist_ok=True)
            with open(args.summary_txt, "w", encoding="utf-8") as f:
                f.write(reporter.build_executive_summary_text())

        out_path = reporter.generate_html(args.out)
    except Exception as e:
        print(f"ERROR generando el reporte: {e}", file=sys.stderr)
        sys.exit(1)

    msg = f"OK: reporte generado en {out_path}"
    if args.summary_txt:
        msg += f" | resumen ejecutivo en {args.summary_txt}"
    print(msg)


if __name__ == "__main__":
    main()
