"""
Data Ingestion Module for E-commerce Customer Intelligence Platform.

Supports loading raw multi-sheet Excel (.xlsx) or CSV files for the
UCI Online Retail II dataset, preserving original data while providing
standardized column mappings and loading metadata.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


# Canonical column names required by specification
EXPECTED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

# Standardized internal naming mapping (safe for SQL and programmatic access)
STANDARDIZED_COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def load_raw_dataset(
    file_path: Union[str, Path],
    sheet_names: Optional[List[str]] = None,
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Load raw transaction records from Excel or CSV.
    
    For Excel files (.xlsx), reads both sheets ('Year 2009-2010' and 'Year 2010-2011')
    unless specific sheets are requested, concatenating them with a sheet origin tag.
    
    Args:
        file_path: Path to .xlsx or .csv dataset file.
        sheet_names: Specific sheets to load if Excel. Default loads all sheets.
        nrows: Optional row limit for sampling or testing.
        
    Returns:
        Tuple of (combined_raw_dataframe, ingestion_metadata_dict).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please run `python scripts/download_dataset.py` to acquire the official dataset."
        )

    metadata = {
        "source_file": str(path.resolve()),
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "file_type": path.suffix.lower(),
        "sheets_loaded": [],
    }

    if path.suffix.lower() == ".xlsx":
        excel_file = pd.ExcelFile(path, engine="openpyxl")
        target_sheets = sheet_names or excel_file.sheet_names
        frames = []
        for sheet in target_sheets:
            df_sheet = pd.read_excel(
                excel_file,
                sheet_name=sheet,
                nrows=nrows,
                dtype={
                    "Invoice": str,
                    "StockCode": str,
                    "Customer ID": "float64",  # accommodates NaNs before Int64 cast
                },
            )
            df_sheet["sheet_source"] = sheet
            frames.append(df_sheet)
            metadata["sheets_loaded"].append(
                {"sheet_name": sheet, "rows_read": len(df_sheet)}
            )
        raw_df = pd.concat(frames, ignore_index=True)

    elif path.suffix.lower() in [".csv", ".txt"]:
        raw_df = pd.read_csv(
            path,
            nrows=nrows,
            dtype={
                "Invoice": str,
                "StockCode": str,
                "Customer ID": "float64",
            },
            low_memory=False,
        )
        raw_df["sheet_source"] = "csv"
        metadata["sheets_loaded"].append(
            {"sheet_name": "csv", "rows_read": len(raw_df)}
        )
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Expected .xlsx or .csv."
        )

    metadata["total_raw_rows"] = len(raw_df)
    metadata["total_raw_columns"] = len(raw_df.columns)
    return raw_df, metadata
