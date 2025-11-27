import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Amazon Paid Ads Performance", layout="wide")

st.markdown(
    """
    <style>
    .main {background-color:#f7f7fb;}
    .block-container {padding-top:1rem;padding-bottom:2rem;}
    h1, h2, h3, h4 {
        font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    }
    .kpi-card {
        padding:0.75rem 1rem;
        border-radius:0.9rem;
        border:1px solid #e5e7eb;
        background:linear-gradient(135deg,#ffffff,#f3f4ff);
        box-shadow:0 8px 16px rgba(15,23,42,0.04);
    }
    .kpi-label {
        font-size:0.75rem;
        color:#6b7280;
        text-transform:uppercase;
        letter-spacing:.08em;
    }
    .kpi-value {
        font-size:1.4rem;
        font-weight:600;
        color:#111827;
    }
    .subtle {
        color:#6b7280;
        font-size:0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align:center;margin-bottom:0.2em;'>Amazon Paid Ads Performance</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subtle' style='text-align:center;'>Analyse croisée Google Search & Meta Ads · Janvier – Juin 2024</p>",
    unsafe_allow_html=True,
)


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


@st.cache_data
def load_and_prepare(path):
    xls = pd.ExcelFile(path)
    google_df = pd.read_excel(xls, sheet_name="google+search")
    meta_df = pd.read_excel(xls, sheet_name="meta+ads")

    google_df = google_df.rename(columns=lambda c: c.strip().lower())
    google_df.rename(
        columns={
            "campaign name": "campaign_name",
            "ad group name": "ad_group_name",
            "conversion rate": "conversion_rate",
        },
        inplace=True,
    )

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

    all_df = pd.concat([google_df, meta_df], ignore_index=True)
    all_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return google_df, meta_df, all_df


google_df, meta_df, all_df = load_and_prepare(
    "data/2025 - Albert School B2 S1 - Digital analytics - Amazon case study.xlsx"
)

st.sidebar.header("Filtres")

platform_filter = st.sidebar.multiselect(
    "Plateforme",
    options=all_df["platform"].unique().tolist(),
    default=all_df["platform"].unique().tolist(),
)

category_filter = st.sidebar.multiselect(
    "Catégorie produit (agrégée)",
    options=sorted(all_df["category_grouped"].dropna().unique().tolist()),
    default=sorted(all_df["category_grouped"].dropna().unique().tolist()),
)

funnel_options = sorted(meta_df["funnel_stage"].dropna().unique().tolist())
funnel_filter = st.sidebar.multiselect(
    "Funnel Meta (TOF/MOF/BOF)",
    options=funnel_options,
    default=funnel_options,
)

df_filtered = all_df[all_df["platform"].isin(platform_filter)]
df_filtered = df_filtered[df_filtered["category_grouped"].isin(category_filter)]

meta_mask = df_filtered["platform"].eq("Meta")
if funnel_filter:
    df_filtered = pd.concat(
        [
            df_filtered[~meta_mask],
            df_filtered[meta_mask & df_filtered["funnel_stage"].isin(funnel_filter)],
        ],
        ignore_index=True,
    )

df_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)

total_spend = df_filtered["spend"].sum()
total_revenue = df_filtered["revenue"].sum()
total_conv = df_filtered["conversions"].sum()
overall_roas = total_revenue / total_spend if total_spend > 0 else np.nan
overall_cpa = total_spend / total_conv if total_conv > 0 else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    f"<div class='kpi-card'><div class='kpi-label'>Dépenses totales</div><div class='kpi-value'>{total_spend:,.0f} €</div></div>",
    unsafe_allow_html=True,
)
c2.markdown(
    f"<div class='kpi-card'><div class='kpi-label'>Revenus totaux</div><div class='kpi-value'>{total_revenue:,.0f} €</div></div>",
    unsafe_allow_html=True,
)
roas_txt = "NA" if np.isnan(overall_roas) else f"{overall_roas:.2f}"
c3.markdown(
    f"<div class='kpi-card'><div class='kpi-label'>ROAS global</div><div class='kpi-value'>{roas_txt}</div></div>",
    unsafe_allow_html=True,
)
cpa_txt = "NA" if np.isnan(overall_cpa) else f"{overall_cpa:.2f} €"
c4.markdown(
    f"<div class='kpi-card'><div class='kpi-label'>CPA global</div><div class='kpi-value'>{cpa_txt}</div></div>",
    unsafe_allow_html=True,
)

tab_q1, tab_q2, tab_q3, tab_q4, tab_q5, tab_q6, tab_q7, tab_adv = st.tabs(
    [
        "Overview & Q1",
        "Q2 – Plateformes & campagnes",
        "Q3 – Catégories",
        "Q4 – Keywords & Ad sets",
        "Q5 – Underperformers",
        "Q6 – CVR vs AOV",
        "Q7 – Budget & Synthèse",
        "Advanced Models",
    ]
)

with tab_q1:
    st.subheader("Q1 – KPIs ajoutés et cadre d’analyse")
    st.markdown(
        """
        Cette vue présente la base de calcul utilisée pour l’ensemble de l’analyse.

        **Pour chaque ligne de campagne (Google ou Meta), les indicateurs suivants sont dérivés :**

        - KPIs média : **CTR**, **CPC**, **conversions**, **CPA**, **ROAS**, **AOV**  
        - Structuration : **plateforme**, **catégorie agrégée** (Tech / Fashion / HomeComfort), **funnel stage** (Search / TOF / MOF / BOF)

        L’objectif est de disposer d’un socle commun permettant de comparer de manière homogène :
        - Google Search vs Meta
        - Campagnes, ad groups, keywords et ad sets
        - Catégories et niveaux de funnel
        """
    )

    cols_to_show = [
        "platform",
        "campaign_name",
        "ad_group_name",
        "ad_set_name",
        "keyword",
        "ad_name",
        "spend",
        "impressions",
        "clicks",
        "conversions",
        "ctr",
        "cpc",
        "cpa",
        "roas",
        "aov",
        "category_grouped",
        "funnel_stage",
    ]
    st.markdown("#### Échantillon de lignes de campagne")
    st.dataframe(df_filtered[cols_to_show].head(30))

    plat_group_global = (
        all_df.groupby("platform")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )
    plat_group_global["roas"] = plat_group_global["revenue"] / plat_group_global[
        "spend"
    ].replace(0, np.nan)
    plat_group_global["cpa"] = plat_group_global["spend"] / plat_group_global[
        "conversions"
    ].replace(0, np.nan)

    st.markdown("#### KPIs globaux par plateforme")
    st.dataframe(
        plat_group_global[["platform", "spend", "revenue", "conversions", "roas", "cpa"]]
    )

    st.markdown(
        """
        Cette table sert de point de départ pour répondre:
        - Q2 :comparaison d’efficacité Google vs Meta
        - Q3 : analyse par catégorie
        - Q7 : recommandation de réallocation budgétaire
        """
    )

with tab_q2:
    st.subheader("Q2 – Performance par plateforme et campagnes leaders")
    st.markdown(
        """
        Cette section répond à la question :  
        **“Quelle plateforme et quelles campagnes délivrent le meilleur arbitrage entre CPA et ROAS ?”**
        """
    )

    plat_group = (
        all_df.groupby("platform")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )
    plat_group["roas"] = plat_group["revenue"] / plat_group["spend"].replace(
        0, np.nan
    )
    plat_group["cpa"] = plat_group["spend"] / plat_group["conversions"].replace(
        0, np.nan
    )

    c21, c22 = st.columns(2)
    fig_roas_plat = px.bar(
        plat_group,
        x="platform",
        y="roas",
        title="ROAS moyen par plateforme",
    )
    c21.plotly_chart(fig_roas_plat, use_container_width=True)

    fig_cpa_plat = px.bar(
        plat_group,
        x="platform",
        y="cpa",
        title="CPA moyen par plateforme",
    )
    c22.plotly_chart(fig_cpa_plat, use_container_width=True)

    google_group = (
        google_df.groupby("campaign_name")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )
    google_group["roas"] = google_group["revenue"] / google_group["spend"].replace(
        0, np.nan
    )
    google_group["cpa"] = google_group["spend"] / google_group["conversions"].replace(
        0, np.nan
    )

    meta_group_campaign = (
        meta_df.groupby("campaign_name")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )
    meta_group_campaign["roas"] = meta_group_campaign["revenue"] / meta_group_campaign[
        "spend"
    ].replace(0, np.nan)
    meta_group_campaign["cpa"] = meta_group_campaign["spend"] / meta_group_campaign[
        "conversions"
    ].replace(0, np.nan)

    g1, g2 = st.columns(2)
    g1.markdown("#### Google – Top campagnes par CPA le plus bas")
    g1.dataframe(
        google_group.sort_values("cpa")[
            ["campaign_name", "spend", "revenue", "conversions", "cpa", "roas"]
        ].head(5)
    )

    g2.markdown("#### Google – Top campagnes par ROAS")
    g2.dataframe(
        google_group.sort_values("roas", ascending=False)[
            ["campaign_name", "spend", "revenue", "conversions", "roas", "cpa"]
        ].head(5)
    )

    m1, m2 = st.columns(2)
    m1.markdown("#### Meta – Top campagnes par CPA le plus bas")
    m1.dataframe(
        meta_group_campaign.sort_values("cpa")[
            ["campaign_name", "spend", "revenue", "conversions", "cpa", "roas"]
        ].head(5)
    )

    m2.markdown("#### Meta – Top campagnes par ROAS")
    m2.dataframe(
        meta_group_campaign.sort_values("roas", ascending=False)[
            ["campaign_name", "spend", "revenue", "conversions", "roas", "cpa"]
        ].head(5)
    )

    st.markdown(
        """
        - Google concentre l’essentiel du chiffre d’affaires et affiche en général le **ROAS le plus élevé**.
        - Meta présente souvent des **CPA plus bas**, particulièrement sur certaines campagnes de retargeting.
        - Les tableaux Top 5 permettent d’identifier les campagnes à **renforcer**, et à l’inverse celles à **réduire ou optimiser**.
        """
    )

with tab_q3:
    st.subheader("Q3 – Performance par catégorie produit (Search vs Social)")
    st.markdown(
        """
        Ici, l’analyse est réalisée au niveau **plateforme × catégorie agrégée** (Tech / Fashion / HomeComfort).

        L’objectif est de comprendre :
        - quelles catégories génèrent la meilleure création de valeur (**ROAS**),
        - quelles catégories permettent d’acquérir un client au **CPA le plus compétitif**,
        - et comment cela diffère entre **Search** et **Social**.
        """
    )

    cat_group = (
        all_df.groupby(["platform", "category_grouped"])
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )
    cat_group["roas"] = cat_group["revenue"] / cat_group["spend"].replace(0, np.nan)
    cat_group["cpa"] = cat_group["spend"] / cat_group["conversions"].replace(0, np.nan)

    fig_roas_cat = px.bar(
        cat_group,
        x="category_grouped",
        y="roas",
        color="platform",
        barmode="group",
        title="ROAS par catégorie et par plateforme",
        labels={"category_grouped": "Catégorie", "roas": "ROAS"},
    )
    st.plotly_chart(fig_roas_cat, use_container_width=True)

    fig_cpa_cat = px.bar(
        cat_group,
        x="category_grouped",
        y="cpa",
        color="platform",
        barmode="group",
        title="CPA par catégorie et par plateforme",
        labels={"category_grouped": "Catégorie", "cpa": "CPA (€)"},
    )
    st.plotly_chart(fig_cpa_cat, use_container_width=True)

    st.markdown(
        """
        - les combinaisons **plateforme × catégorie** à **ROAS élevé et CPA contenu** (candidats à la montée en budget),
        - et celles à **ROAS faible / CPA élevé** (zones à optimiser ou à désinvestir).
        """
    )

with tab_q4:
    st.subheader("Q4 – Top keywords Google et Top ad sets Meta (ROAS)")
    st.markdown(
        """
        Cette section isole les **leviers les plus efficaces** :

        - côté Google : **keywords** qui concentrent un ROAS très élevé,
        - côté Meta : **ad sets** les plus performants en ROAS (et souvent BOF / retargeting).
        """
    )

    g_kw = (
        google_df.groupby("keyword")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )
    g_kw["roas"] = g_kw["revenue"] / g_kw["spend"].replace(0, np.nan)
    g_kw["cpa"] = g_kw["spend"] / g_kw["conversions"].replace(0, np.nan)

    m_sets = (
        meta_df.groupby("ad_set_name")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )
    m_sets["roas"] = m_sets["revenue"] / m_sets["spend"].replace(0, np.nan)
    m_sets["cpa"] = m_sets["spend"] / m_sets["conversions"].replace(0, np.nan)

    k1, k2 = st.columns(2)
    k1.markdown("#### Google – Top 5 keywords par ROAS")
    k1.dataframe(
        g_kw.sort_values("roas", ascending=False)[
            ["keyword", "spend", "revenue", "conversions", "roas", "cpa"]
        ].head(5)
    )

    k2.markdown("#### Meta – Top 5 ad sets par ROAS")
    k2.dataframe(
        m_sets.sort_values("roas", ascending=False)[
            ["ad_set_name", "spend", "revenue", "conversions", "roas", "cpa"]
        ].head(5)
    )

    st.markdown(
        """
        - sur Google, les keywords top ROAS correspondent généralement à des requêtes très intentionnelles sur des produits Tech/Fashion à fort AOV ;
        - sur Meta, les ad sets top ROAS sont majoritairement des audiences BOF (cart visitors, cart abandoners, past buyers), directement liées à la conversion.
        """
    )

with tab_q5:
    st.subheader("Q5 – Underperformers et hypothèses d’underperformance")
    st.markdown(
        """
        Cette section met en avant les **pire performeurs** en termes de ROAS, côté Google et Meta.  
        Ce sont les candidats naturels pour :
        - un **diagnostic ciblé** (ciblage, enchères, créas),
        - voire un **désinvestissement** si la marge ne justifie pas l’effort.
        """
    )

    worst_google = google_group.sort_values("roas", ascending=True).head(5)
    worst_meta_sets = m_sets.sort_values("roas", ascending=True).head(5)

    w1, w2 = st.columns(2)
    w1.markdown("#### Bottom 5 campagnes Google par ROAS")
    w1.dataframe(
        worst_google[
            ["campaign_name", "spend", "revenue", "conversions", "roas", "cpa"]
        ]
    )

    w2.markdown("#### Bottom 5 ad sets Meta par ROAS")
    w2.dataframe(
        worst_meta_sets[
            ["ad_set_name", "spend", "revenue", "conversions", "roas", "cpa"]
        ]
    )

    st.markdown(
        """
        - **ciblage trop large** ou trop haut de funnel (TOF) → peu d’intent d’achat immédiat ;
        - **enchères élevées** sur des segments très concurrentiels → CPC et CPA explosent ;
        - **créations peu différenciantes** sur des produits commoditisés ;
        - **produits à très fort panier moyen** : taux de conversion plus faible, d’où un ROAS dégradé à budget constant.
        """
    )

with tab_q6:
    st.subheader("Q6 – Relation CVR vs AOV et interprétation par plateforme")
    st.markdown(
        """
        - évaluer si les **catégories à haut AOV** obtiennent mécaniquement un meilleur ROAS ;
        - comparer la dynamique **Search vs Social** en termes d’intent utilisateur.
        """
    )

    c61, c62 = st.columns(2)

    g_cvr_aov = google_df[["conv_rate_pct", "aov"]].dropna()
    if len(g_cvr_aov) >= 2:
        xg = g_cvr_aov["conv_rate_pct"].values
        yg = g_cvr_aov["aov"].values
        mg, bg = np.polyfit(xg, yg, 1)
        rg = np.corrcoef(xg, yg)[0, 1]
        xs_g = np.linspace(xg.min(), xg.max(), 100)
        ys_g = mg * xs_g + bg
        fig_g = px.scatter(
            g_cvr_aov,
            x="conv_rate_pct",
            y="aov",
            title="Google : CVR (%) vs AOV (€)",
            labels={"conv_rate_pct": "CVR (%)", "aov": "AOV (€)"},
        )
        fig_g.add_trace(
            go.Scatter(x=xs_g, y=ys_g, mode="lines", name="Régression linéaire")
        )
        c61.plotly_chart(fig_g, use_container_width=True)
        c61.metric("Corrélation CVR–AOV (Google)", f"{rg:.2f}")
    else:
        c61.write("Données insuffisantes pour la régression Google.")

    m_cvr_aov = meta_df[["cvr_pct", "aov"]].dropna()
    if len(m_cvr_aov) >= 2:
        xm = m_cvr_aov["cvr_pct"].values
        ym = m_cvr_aov["aov"].values
        mm_l, bm_l = np.polyfit(xm, ym, 1)
        rm = np.corrcoef(xm, ym)[0, 1]
        xs_m = np.linspace(xm.min(), xm.max(), 100)
        ys_m = mm_l * xs_m + bm_l
        fig_m = px.scatter(
            m_cvr_aov,
            x="cvr_pct",
            y="aov",
            title="Meta : CVR (%) vs AOV (€)",
            labels={"cvr_pct": "CVR (%)", "aov": "AOV (€)"},
        )
        fig_m.add_trace(
            go.Scatter(x=xs_m, y=ys_m, mode="lines", name="Régression linéaire")
        )
        c62.plotly_chart(fig_m, use_container_width=True)
        c62.metric("Corrélation CVR–AOV (Meta)", f"{rm:.2f}")
    else:
        c62.write("Données insuffisantes pour la régression Meta.")

    st.markdown(
        """
        - sur **Google**, un AOV élevé s’accompagne souvent d’un **CVR plus faible** (parcours d’achat plus long, plus de comparaison) ;
        - sur **Meta**, la relation est souvent plus diffuse : le canal joue un rôle de **retargeting** et d’**amplification** plutôt que de décision pure.

        un **AOV élevé** ne garantit **pas** automatiquement un **ROAS élevé**, surtout si les coûts médias (CPC, CPA) s’ajustent à la hausse.
        """
    )

with tab_q7:
    st.subheader("Q7 – Synthèse")
    plat_global = (
        all_df.groupby("platform")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )
    plat_global["roas"] = plat_global["revenue"] / plat_global["spend"].replace(
        0, np.nan
    )
    plat_global["cpa"] = plat_global["spend"] / plat_global["conversions"].replace(
        0, np.nan
    )

    st.markdown("#### KPIs globaux de référence")
    st.dataframe(
        plat_global[["platform", "spend", "revenue", "conversions", "roas", "cpa"]]
    )

    st.markdown(
        """
        - **Google Search**  
          - Canal principal de création de valeur : ROAS globalement supérieur, volumétrie de revenus majeure.  
          - Renforcer les **campagnes et mots-clés à fort ROAS** sur Tech/Fashion.  
          - Rationaliser les campagnes HomeComfort aux **CPA très élevés**.

        - **Meta (Facebook / Instagram)**  
          - Rôle clé en **bas de funnel** (BOF) : retargeting de visiteurs et acheteurs existants à CPA compétitifs.  
          - Focaliser les budgets sur les **ad sets BOF** les plus performants.  
          - Réduire ou repositionner les campagnes **TOF peu rentables**, en les évaluant plutôt sur des KPIs de reach/engagement.

        - **Mix global Search + Social**  
          - Positionner Google comme **générateur d’intent et de trafic qualifié**,  
          - Utiliser Meta comme **levier de conversion et de réactivation** sur les audiences chaudes.  

        - ROAS et CPA par plateforme, catégorie et funnel,
        - top & flop campagnes / ad sets,
        - corrélation entre investissements et revenus.
        """
    )

with tab_adv:
    st.subheader("Advanced Models – Drivers, corrélations et segmentation")
    st.markdown(
        """
        - compréhension des **corrélations** entre KPIs,
        - modélisation de la relation **Revenue ~ variables médias**,
        - **segmentation automatique** des campagnes/ad sets en clusters actionnables.
        """
    )

    num_cols = [
        "spend",
        "impressions",
        "clicks",
        "conversions",
        "ctr",
        "cpc",
        "cpa",
        "roas",
        "aov",
    ]
    if "cpm_eur" in all_df.columns:
        num_cols.append("cpm_eur")

    corr_df = all_df[num_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(corr_df) > 2:
        corr = corr_df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            title="Corrélation entre KPIs (tt plateformes)",
            aspect="auto",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.write("Données insuffisantes pour une heatmap de corrélation.")

    st.markdown("#### Régression multivariée : Revenue ~ Spend + Impressions + Clicks + Dummies")

    regr_df = all_df[
        ["spend", "impressions", "clicks", "platform", "category_grouped", "funnel_stage", "revenue"]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    if len(regr_df) >= 20:
        dummies = pd.get_dummies(
            regr_df[["platform", "category_grouped", "funnel_stage"]], drop_first=True
        )
        X = pd.concat([regr_df[["spend", "impressions", "clicks"]], dummies], axis=1)
        y = regr_df["revenue"].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_full = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        fig_pred = px.scatter(
            x=y,
            y=y_pred,
            labels={"x": "Revenue réel", "y": "Revenue prédit"},
            title="Régression Revenue – Réel vs prédit",
        )
        fig_pred.add_trace(
            go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode="lines", name="Idéal")
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        st.metric("R² de la régression Revenue", f"{r2_full:.2f}")

        coef_df = pd.DataFrame({"feature": X.columns, "coef": model.coef_})
        coef_df["abs_coef"] = coef_df["coef"].abs()
        st.markdown("Top drivers (coefficients en valeur absolue)")
        st.dataframe(coef_df.sort_values("abs_coef", ascending=False).head(10)[["feature", "coef"]])
    else:
        st.write("Données insuffisantes pour une régression fiable.")

    st.markdown("#### Segmentation K-Means des campagnes / ad sets (ROAS – CPA – CTR – CPC)")

    g_units = (
        google_df.groupby("campaign_name")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
        )
        .reset_index()
    )
    g_units["platform"] = "Google"
    g_units["unit_name"] = g_units["campaign_name"]
    g_units["ctr"] = g_units["clicks"] / g_units["impressions"]
    g_units["cpc"] = g_units["spend"] / g_units["clicks"].replace(0, np.nan)
    g_units["cpa"] = g_units["spend"] / g_units["conversions"].replace(0, np.nan)
    g_units["roas"] = g_units["revenue"] / g_units["spend"].replace(0, np.nan)

    m_units = (
        meta_df.groupby("ad_set_name")
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            conversions=("conversions", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
        )
        .reset_index()
    )
    m_units["platform"] = "Meta"
    m_units["unit_name"] = m_units["ad_set_name"]
    m_units["ctr"] = m_units["clicks"] / m_units["impressions"]
    m_units["cpc"] = m_units["spend"] / m_units["clicks"].replace(0, np.nan)
    m_units["cpa"] = m_units["spend"] / m_units["conversions"].replace(0, np.nan)
    m_units["roas"] = m_units["revenue"] / m_units["spend"].replace(0, np.nan)

    perf_units = pd.concat([g_units, m_units], ignore_index=True)
    perf_units = perf_units.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["roas", "cpa", "ctr", "cpc"]
    )

    if len(perf_units) >= 8:
        X_clust = perf_units[["roas", "cpa", "ctr", "cpc"]].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clust)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        perf_units["cluster"] = kmeans.fit_predict(X_scaled)

        med_roas = perf_units["roas"].median()
        med_cpa = perf_units["cpa"].median()
        segments = []
        for _, r in perf_units.iterrows():
            if r["roas"] >= med_roas and r["cpa"] <= med_cpa:
                segments.append("HERO (Scale Up)")
            elif r["roas"] >= med_roas and r["cpa"] > med_cpa:
                segments.append("Premium (Selective)")
            elif r["roas"] < med_roas and r["cpa"] <= med_cpa:
                segments.append("Support / Awareness")
            else:
                segments.append("Fix or Kill")
        perf_units["segment"] = segments

        fig_seg = px.scatter(
            perf_units,
            x="cpa",
            y="roas",
            color="segment",
            symbol="platform",
            hover_data=["unit_name", "spend", "revenue", "conversions"],
            title="Frontière ROAS–CPA par segment de performance",
            labels={"cpa": "CPA (€)", "roas": "ROAS"},
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        st.markdown("Tableau de segmentation détaillé")
        st.dataframe(
            perf_units[
                [
                    "platform",
                    "unit_name",
                    "spend",
                    "revenue",
                    "conversions",
                    "roas",
                    "cpa",
                    "ctr",
                    "cpc",
                    "segment",
                ]
            ]
            .sort_values(["segment", "roas"], ascending=[True, False])
            .reset_index(drop=True)
        )

        st.markdown(
            """
            - **HERO (Scale Up)** : campagnes / ad sets à scaler en priorité (ROAS élevé, CPA maîtrisé) ;
            - **Premium (Selective)** : performants mais coûteux, à réserver à des moments / audiences clés ;
            - **Support / Awareness** : utiles pour alimenter le funnel, à piloter avec des KPIs de reach et de fréquence ;
            - **Fix or Kill** : leviers à corriger rapidement ou à arrêter.
            """
        )
    else:
        st.write("Données insuffisantes pour une segmentation K-Means.")
