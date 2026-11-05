"""Display formatting only; stored tables and model identifiers remain unchanged."""
import pandas as pd
from IPython.display import display

LABELS = {
    "FT-MDN": "FT-MDN-T", "FT-Transformer-MDN": "FT-MDN-T",
    "FT–MDN–T": "FT-MDN-T", "FT-Reg": "FT-Reg", "XGB": "XGBoost",
    "scratch": "FT-MDN-T (scratch)", "transfer": "FT-MDN-T (pretrained)",
    "ridge": "Gaussian ridge",
}


def present_table(value):
    """Use four significant digits, retaining small diagnostics and integer counts."""
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if not isinstance(value, pd.DataFrame):
        display(value)
        return
    table = value.copy()
    table = table.rename(index=LABELS, columns=LABELS)
    table.columns = [col.replace("FT_", "FT-MDN-T_") if isinstance(col, str) else col
                     for col in table.columns] if not isinstance(table.columns, pd.MultiIndex) else table.columns
    for col in table.select_dtypes(include=["object", "string"]):
        table[col] = table[col].replace(LABELS)
    formats = {col: "{:.4g}" for col in table.select_dtypes(include="floating")}
    display(table.style.format(formats, na_rep="—"))
