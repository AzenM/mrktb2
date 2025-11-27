import pandas as pd
import numpy as np
import datetime
from pathlib import Path

DATA_PATH = Path("data")
INPUT_XLSX = DATA_PATH / "2025 - Albert School B2 S1 - Digital analytics - Amazon case study.xlsx"
OUTPUT_GOOGLE = DATA_PATH / "google_enriched.csv"
OUTPUT_META = DATA_PATH / "meta_enriched.csv"


def parse_rate_cell(val):
    if isinstance(val, (int, float, np.floating)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except Exception:
            return np.nan
    if isinstance(val, datetime.datetime):
        return float(val.day)
    return np.nan


def parse_cpm_cell(val):
    if isinstance(val, (int, float, np.floating)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except Exception:
            return np.nan
    if isinstance(val, datetime.datetime):
        return float(val.month + val.day / 100.0)
    return np.nan


def detect_category_meta(name: str) -> str:
    name = str(name)
    if "Tech" in name or "tech" in name:
        return "Tech"
    if "Fashion" in name or "fashion" in name or "Clothes" in name or "clothes" in name:
        return "Fashion"
    if "Home Comfort" in name or "Home" in name or "Appliances" in name or "Decor" in name:
        return "HomeComfort"
    return "Other"


def map_category_search(campaign_name: str) -> str:
    if not isinstance(campaign_name, str):
        return "Other"
    root = campaign_name.split(" - ")[0]
    if root.startswith("Tech"):
        return "Tech"
    if root.startswith("Clothes") or root.startswith("Fashion"):
        return "Fashion"
    if root.startswith("Home"):
        return "HomeComfort"
    return "Other"


def load_and_enrich():
    xls = pd.ExcelFile(INPUT_XLSX)
    google_df = pd.read_excel(xls, sheet_name="google+search")
    meta_df = pd.read_excel(xls, sheet_name="meta+ads")

    google_df = google_df.rename(columns=lambda c: c.strip().lower())
    google_df.rename(columns={
        "campaign name": "campaign_name",
        "ad group name": "ad_group_name",
        "conversion rate": "conversion_rate",
    }, inplace=True)

    for col in ["spend", "impressions", "clicks", "revenue"]:
        if col in google_df.columns:
            google_df[col] = pd.to_numeric(google_df[col], errors="coerce")

    google_df["conv_rate_pct"] = google_df["conversion_rate"].apply(parse_rate_cell)

    google_df["platform"] = "Google"
    google_df["product_category"] = google_df["campaign_name"].str.split(" - ").str[0]
    google_df["category_grouped"] = google_df["campaign_name"].apply(map_category_search)

    google_df["ctr"] = google_df["clicks"] / google_df["impressions"]
    google_df["cpc"] = google_df["spend"] / google_df["clicks"].replace(0, np.nan)
    google_df["conversions"] = google_df["clicks"] * (google_df["conv_rate_pct"] / 100.0)
    google_df["cpa"] = google_df["spend"] / google_df["conversions"].replace(0, np.nan)
    google_df["roas"] = google_df["revenue"] / google_df["spend"].replace(0, np.nan)
    google_df["aov"] = google_df["revenue"] / google_df["conversions"].replace(0, np.nan)
    google_df["funnel_stage"] = "Search"
    google_df["ad_set_name"] = np.nan
    google_df["ad_name"] = google_df["keyword"]

    meta_df = meta_df.rename(columns=lambda c: c.strip().lower())

    for col in ["spend", "impressions", "reach", "clicks", "aov"]:
        if col in meta_df.columns:
            meta_df[col] = pd.to_numeric(meta_df[col], errors="coerce")

    meta_df["cvr_pct"] = meta_df["cvr"].apply(parse_rate_cell)
    meta_df["cpm_eur"] = meta_df["cpm"].apply(parse_cpm_cell)

    meta_df["platform"] = "Meta"
    meta_df["funnel_stage"] = meta_df["campaign_name"].str.split(" - ").str[0]
    meta_df["product_category"] = meta_df["campaign_name"].apply(detect_category_meta)
    meta_df["category_grouped"] = meta_df["product_category"].apply(detect_category_meta)

    meta_df["ctr"] = meta_df["clicks"] / meta_df["impressions"]
    meta_df["cpc"] = meta_df["spend"] / meta_df["clicks"].replace(0, np.nan)
    meta_df["conversions"] = meta_df["clicks"] * (meta_df["cvr_pct"] / 100.0)
    meta_df["revenue"] = meta_df["conversions"] * meta_df["aov"]
    meta_df["cpa"] = meta_df["spend"] / meta_df["conversions"].replace(0, np.nan)
    meta_df["roas"] = meta_df["revenue"] / meta_df["spend"].replace(0, np.nan)

    return google_df, meta_df


def main():
    DATA_PATH.mkdir(exist_ok=True)
    google_df, meta_df = load_and_enrich()

    google_df.to_csv(OUTPUT_GOOGLE, index=False)
    meta_df.to_csv(OUTPUT_META, index=False)

    print("Enriched Google data ->", OUTPUT_GOOGLE)
    print("Enriched Meta data   ->", OUTPUT_META)

    g_campaign = google_df.groupby("campaign_name").agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        conversions=("conversions", "sum")
    ).reset_index()
    g_campaign["cpa"] = g_campaign["spend"] / g_campaign["conversions"]
    g_campaign["roas"] = g_campaign["revenue"] / g_campaign["spend"]

    m_campaign = meta_df.groupby("campaign_name").agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        conversions=("conversions", "sum")
    ).reset_index()
    m_campaign["cpa"] = m_campaign["spend"] / m_campaign["conversions"]
    m_campaign["roas"] = m_campaign["revenue"] / m_campaign["spend"]

    print("\nTop 3 Google campaigns by ROAS:")
    print(g_campaign.sort_values("roas", ascending=False).head(3))

    print("\nTop 3 Meta campaigns by ROAS:")
    print(m_campaign.sort_values("roas", ascending=False).head(3))


if __name__ == "__main__":
    main()
