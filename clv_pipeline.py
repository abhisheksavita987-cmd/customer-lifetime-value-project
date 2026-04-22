import pandas as pd
import numpy as np
import sqlite3
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sqlalchemy import create_engine


# Adjust these paths before running
DATASET_PATH  = "Online Retail.xlsx"          # raw data (Excel)
DB_PATH       = "outputs/ecommerce_clv.db"    # SQLite database
PLOTS_DIR     = "plots"                        # folder for saved figures
CSV_CLEANED   = "outputs/cleaned_data.csv"
CSV_RFM       = "outputs/rfm_table.csv"
CSV_SEGMENTS  = "outputs/customer_segments.csv"

os.makedirs("outputs", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """
    Load the Online Retail dataset from an Excel file.
    Supports both .xlsx and .csv input.
    """
    print("\n[1/7] Loading raw dataset ...")
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path, dtype={"CustomerID": str})
    else:
        df = pd.read_csv(path, dtype={"CustomerID": str}, encoding="ISO-8859-1")

    print(f"      Raw rows : {len(df):,}")
    print(f"      Columns  : {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataframe:
      - Remove cancelled orders  (InvoiceNo starts with 'C')
      - Remove rows with missing CustomerID
      - Remove rows with Quantity <= 0
      - Remove rows with UnitPrice <= 0
      - Convert InvoiceDate to datetime
      - Add TotalPrice = Quantity * UnitPrice
    """
    print("\n[2/7] Cleaning data ...")

    # Step 2a – Remove cancelled orders
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    print(f"      After removing cancellations : {len(df):,} rows")

    # Step 2b – Drop rows with null CustomerID
    df = df.dropna(subset=["CustomerID"])
    print(f"      After dropping null CustID   : {len(df):,} rows")

    # Step 2c – Remove invalid Quantity (≤ 0)
    df = df[df["Quantity"] > 0]
    print(f"      After removing bad Quantity  : {len(df):,} rows")

    # Step 2d – Remove invalid UnitPrice (≤ 0)
    df = df[df["UnitPrice"] > 0]
    print(f"      After removing bad UnitPrice : {len(df):,} rows")

    # df.to_sql('cleaned_data', con=engine, if_exists='replace', index=False)  # Disabled
    # print("Data uploaded to MySQL")
    print("Skipped MySQL upload - using SQLite")
    
    # Step 2e – Convert date column
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Step 2f – Clean CustomerID (strip decimals if float-encoded)
    df["CustomerID"] = df["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    # Step 2g – Derived column: TotalPrice per line item
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    print(f"      Final cleaned rows           : {len(df):,}")
    return df


def store_in_db(df: pd.DataFrame, db_path: str) -> sqlite3.Connection:
    """
    Creates an SQLite database and stores the cleaned transactions.
    The CREATE TABLE + INSERT statements are identical to MySQL syntax.
    """
    print("\n[3/7] Storing cleaned data in SQLite (MySQL-compatible) ...")

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # MySQL-compatible CREATE TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            InvoiceNo   TEXT,
            StockCode   TEXT,
            Description TEXT,
            Quantity    INTEGER,
            InvoiceDate TEXT,
            UnitPrice   REAL,
            CustomerID  TEXT,
            Country     TEXT,
            TotalPrice  REAL
        )
    """)
    conn.commit()

    # Bulk-insert cleaned data
    df_db = df.copy()
    df_db["InvoiceDate"] = df_db["InvoiceDate"].astype(str)
    df_db.to_sql("transactions", conn, if_exists="replace", index=False)

    row_count = cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    print(f"      Rows inserted into DB : {row_count:,}")
    return conn


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Recency, Frequency, Monetary for each customer.

    Recency  : Days since last purchase  (lower = more recent)
    Frequency: Number of distinct invoices
    Monetary : Total money spent
    """
    print("\n[4/7] Computing RFM features ...")

    # Reference date = one day after the latest transaction
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency = ("InvoiceNo",   "nunique"),
        Monetary  = ("TotalPrice",  "sum")
    ).reset_index()

    print(f"      Unique customers : {len(rfm):,}")
    print(rfm.describe().round(2))
    return rfm


def train_clv_model(rfm: pd.DataFrame):
    """
    Trains a Linear Regression model to predict CLV (Monetary value).

    Features (X) : Recency, Frequency   (normalised)
    Target  (y)  : Monetary             (log-transformed for better fit)

    Returns the trained model, scaler, rfm with predictions, and metrics dict.
    """
    print("\n[5/7] Training Linear Regression model ...")

    # 5a – Create target:  log(Monetary + 1)  to reduce skew
    rfm = rfm.copy()
    rfm["CLV_actual"] = np.log1p(rfm["Monetary"])

    # 5b – Feature matrix
    X = rfm[["Recency", "Frequency"]].copy()
    y = rfm["CLV_actual"]

    # 5c – Normalise features with MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 5d – Train / test split  (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 5e – Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5f – Predict on test set
    y_pred = model.predict(X_test)

    # 5g – Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    metrics = {"MAE": round(mae, 4), "MSE": round(mse, 4), "R²": round(r2, 4)}
    print(f"      MAE : {mae:.4f}")
    print(f"      MSE : {mse:.4f}")
    print(f"      R²  : {r2:.4f}")

    # 5h – Predict CLV for ALL customers and back-transform
    rfm["CLV_predicted_log"] = model.predict(scaler.transform(X))
    rfm["CLV_predicted"]     = np.expm1(rfm["CLV_predicted_log"]).clip(lower=0)

    return model, scaler, rfm, metrics, y_test, y_pred


def segment_customers(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Segments customers into 5 groups based on RFM + predicted CLV:

        VIP       – High frequency, high monetary, recent
        Loyal     – Moderate frequency & monetary, recent
        Potential – Low frequency but recent and some monetary value
        At-Risk   – Haven't purchased recently but had decent value
        Lost      – Very old last purchase, low frequency & monetary
    """
    print("\n[6/7] Segmenting customers ...")

    # Create RFM score percentile ranks (1–5, 5 = best)
    rfm["R_score"] = pd.qcut(rfm["Recency"],   q=5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    # Rule-based segmentation
    def assign_segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "VIP"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyal"
        elif r >= 3 and f <= 2:
            return "Potential"
        elif r <= 2 and (f >= 3 or m >= 3):
            return "At-Risk"
        else:
            return "Lost"

    rfm["Segment"] = rfm.apply(assign_segment, axis=1)

    seg_counts = rfm["Segment"].value_counts()
    print("\n      Customer Segment Counts:")
    for seg, cnt in seg_counts.items():
        print(f"        {seg:<12}: {cnt:,}")

    return rfm


def generate_plots(rfm: pd.DataFrame, y_test, y_pred, plots_dir: str):
    """
    Generates 5 plots required for the project:
      Fig 1 – RFM Distribution Histograms
      Fig 2 – Customer Segmentation Pie Chart
      Fig 3 – Predicted vs Actual CLV  (scatter line)
      Fig 4 – Correlation Heatmap of RFM features
      Fig 5 – Top 10 Customers by Predicted CLV
    """
    print(f"\n[7/7] Generating visualisations → {plots_dir}/")
    sns.set_theme(style="whitegrid", palette="muted")

    # ── Fig 1 : RFM Histograms ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("RFM Distribution Histograms", fontsize=14, fontweight="bold")

    rfm["Recency"].plot.hist(bins=40, ax=axes[0], color="#4C72B0", edgecolor="white")
    axes[0].set_title("Recency (days)")
    axes[0].set_xlabel("Days since last purchase")

    rfm["Frequency"].clip(upper=rfm["Frequency"].quantile(0.99)).plot.hist(
        bins=40, ax=axes[1], color="#DD8452", edgecolor="white")
    axes[1].set_title("Frequency (invoices)")
    axes[1].set_xlabel("Number of transactions")

    rfm["Monetary"].clip(upper=rfm["Monetary"].quantile(0.99)).plot.hist(
        bins=40, ax=axes[2], color="#55A868", edgecolor="white")
    axes[2].set_title("Monetary (£)")
    axes[2].set_xlabel("Total spend (£)")

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/fig1_rfm_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: fig1_rfm_histograms.png")

    # ── Fig 2 : Segmentation Pie Chart ───────────────────────────────────────
    seg_counts = rfm["Segment"].value_counts()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9E9E9E"]
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        seg_counts.values,
        labels=seg_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Customer Segmentation\n(RFM-based)", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/fig2_customer_segments_pie.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: fig2_customer_segments_pie.png")

    # ── Fig 3 : Predicted vs Actual CLV ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    y_test_arr = np.array(y_test)
    sorted_idx = np.argsort(y_test_arr)
    ax.plot(y_test_arr[sorted_idx],  label="Actual CLV (log)",    color="#1565C0", linewidth=1.5)
    ax.plot(y_pred[sorted_idx],      label="Predicted CLV (log)", color="#E53935", linewidth=1.5, linestyle="--")
    ax.set_title("Predicted vs Actual CLV (Test Set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Customer index (sorted by actual)")
    ax.set_ylabel("CLV  (log scale)")
    ax.legend()
    ax.fill_between(range(len(sorted_idx)), y_test_arr[sorted_idx], y_pred[sorted_idx],
                    alpha=0.15, color="orange", label="Residual")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/fig3_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: fig3_predicted_vs_actual.png")

    # ── Fig 4 : Correlation Heatmap ───────────────────────────────────────────
    corr_cols = ["Recency", "Frequency", "Monetary", "CLV_predicted", "R_score", "F_score", "M_score"]
    corr_data = rfm[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, square=True)
    ax.set_title("Correlation Heatmap – RFM & CLV Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/fig4_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: fig4_correlation_heatmap.png")

    # ── Fig 5 : Top 10 Customers by CLV ──────────────────────────────────────
    top10 = rfm.nlargest(10, "CLV_predicted")[["CustomerID", "CLV_predicted", "Segment"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top10["CustomerID"].astype(str), top10["CLV_predicted"],
                   color=["#2196F3" if s == "VIP" else "#4CAF50" for s in top10["Segment"]],
                   edgecolor="white")
    ax.set_xlabel("Predicted CLV (£)")
    ax.set_title("Top 10 Customers by Predicted CLV", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, top10["CLV_predicted"]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"£{val:,.0f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/fig5_top10_customers.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: fig5_top10_customers.png")


def main():
    print("=" * 60)
    print("  CLV Forecasting Pipeline ")
    print("=" * 60)

    # Stage 1 – Load
    df_raw = load_data(DATASET_PATH)

    # Stage 2 – Clean
    df_clean = clean_data(df_raw)
    df_clean.to_csv(CSV_CLEANED, index=False)

    # Stage 3 – Store in DB
    conn = store_in_db(df_clean, DB_PATH)

    # Stage 4 – RFM
    rfm = compute_rfm(df_clean)
    rfm.to_csv(CSV_RFM, index=False)

    # Stage 5 – ML Model
    model, scaler, rfm, metrics, y_test, y_pred = train_clv_model(rfm)

    # Stage 6 – Segmentation
    rfm = segment_customers(rfm)
    rfm.to_csv(CSV_SEGMENTS, index=False)

    # Stage 7 – Plots
    generate_plots(rfm, y_test, y_pred, PLOTS_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Model Performance:")
    for k, v in metrics.items():
        print(f"    {k} : {v}")

    print(f"\n  Output files:")
    print(f"    {CSV_CLEANED}   → cleaned transactions")
    print(f"    {CSV_RFM}       → RFM scores per customer")
    print(f"    {CSV_SEGMENTS}  → segments + CLV predictions")
    print(f"    {DB_PATH}       → SQLite database (Power BI source)")
    print(f"    {PLOTS_DIR}/    → all 5 figures")
    print("\n  Import 'outputs/customer_segments.csv' into Power BI\n")


if __name__ == "__main__":
    main()
