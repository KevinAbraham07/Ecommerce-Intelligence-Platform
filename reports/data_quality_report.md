# Data Quality & Ingestion Audit Report
## UCI Online Retail II Dataset

### 1. Ingestion Overview
- **Source File**: `C:\Users\kevin\Ecomm Intelli PF\data\raw\online_retail_II.xlsx`
- **File Size**: `43.51 MB`
- **Total Rows Read**: `1,067,371`
- **Date Range**: `2009-12-01 07:45:00` to `2011-12-09 12:50:00` (738 days)
- **Sheets Loaded**:
  - `Year 2009-2010`: 525,461 rows
  - `Year 2010-2011`: 541,910 rows

### 2. High-Level Entity Counts
- **Unique Invoices**: `53,628`
- **Unique Customers (non-null)**: `5,942`
- **Unique Stock Codes**: `5,305`
- **Unique Countries**: `43`

### 3. Missing Value Analysis
| Column | Missing Count | Missing % | Data Type |
|---|---|---|---|
| Customer ID | 243,007 | 22.77% | float64 |
| Description | 4,382 | 0.41% | object |
| Invoice | 0 | 0.0% | str |
| Quantity | 0 | 0.0% | int64 |
| StockCode | 0 | 0.0% | str |
| InvoiceDate | 0 | 0.0% | datetime64[us] |
| Price | 0 | 0.0% | float64 |
| Country | 0 | 0.0% | str |
| sheet_source | 0 | 0.0% | str |

> **Key Finding**: `Customer ID` is missing in `243,007` rows (22.77%). 
> Guest / guest-checkout transactions must not be blindly imputed or discarded for macro sales trends, but must be segregated for customer-level analytics (RFM, cohorts, repeat purchase prediction).

### 4. Integrity & Anomaly Audit
- **Duplicate Rows**: `34,335` (3.22%)
- **Cancellations (Invoice starting with 'C')**: `19,494` (1.83%)
- **Negative Quantities**: `22,950`
- **Negative Quantities without 'C' prefix**: `3,457` (damages, inventory write-offs, manual corrections)
- **Zero Quantities**: `0`
- **Zero Unit Prices**: `6,202`
- **Negative Unit Prices**: `5` (e.g. bad debt adjustments)
- **Administrative / Non-Product Stock Codes**: `5,525` (e.g. POST, D, M, BANK CHARGES)

### 5. Cleaning Recommendations for Downstream Modules
1. **Gross vs. Net Revenue Policy**:
   - Gross sales: Positive quantity and positive price transactions.
   - Net sales: Include cancellations/returns, or net against parent purchases where applicable.
2. **Market-Basket Analysis Policy**:
   - Must exclude cancellations and non-product stock codes (`POST`, `D`, `M`, etc.) to prevent false co-occurrences.
3. **Customer Analytics Policy (RFM & Retention)**:
   - Strictly evaluate customers with valid `Customer ID`.
   - Returns should offset monetary spend accurately without generating negative lifetime customer values.
