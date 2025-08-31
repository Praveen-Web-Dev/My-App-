# app.py (asset-free, descriptive, with "Exclude 2020" option)
# -----------------------------------------------------------
# Global Energy, Renewables & SDGs – Interactive Dashboard
# -----------------------------------------------------------
# pip install streamlit pandas numpy plotly statsmodels

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# import statsmodels.api as sm
import streamlit as st

# ---------------------------
# Page & light theming
# ---------------------------
st.set_page_config(page_title="Global Energy, Renewables & SDGs", layout="wide", page_icon="🌍")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
/* Hero card */
.hero {
  border-radius: 16px;
  padding: 18px 22px;
  background: linear-gradient(135deg, rgba(41,121,255,0.08), rgba(38,166,154,0.08));
  border: 1px solid rgba(0,0,0,0.05);
}
/* KPI cards */
.kpi-card {
  border-radius: 14px; padding: 16px;
  background: #ffffff;
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}
/* Small caption text */
.small { font-size: 0.9rem; color: #5c6773; }
/* Legend/help chips */
.help-chip {
  display: inline-block; padding: 6px 10px; margin: 6px 4px 0 0;
  border-radius: 999px; background: #f4f6f8; border: 1px solid #e6e9ec;
  font-size: 0.85rem; color: #415266;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
def load_data_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def rename_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'entity': 'country',
        'region': 'region',
        'income_group': 'income',
        'year': 'year',
        'access_to_electricity_(%_of_population)': 'electricity_access',
        'renewable_energy_share_in_the_total_final_energy_consumption_(%)': 'renewable_share',
        'value_co2_emissions_kt_by_country': 'co2_kt',
        'primary_energy_consumption_per_capita_(kwh/person)': 'energy_per_capita',
        'energy_intensity_level_of_primary_energy_(mj/$2017_ppp_gdp)': 'energy_intensity',
        'gdp_per_capita': 'gdp_pc',
        'gdp_growth': 'gdp_growth',
        'financial_flows_to_developing_countries_(us_$)': 'climate_finance',
        'electricity_from_fossil_fuels_(twh)': 'elec_fossil_twh',
        'electricity_from_renewables_(twh)': 'elec_ren_twh',
        'low-carbon_electricity_(%_electricity)': 'low_carbon_pct',
        'latitude': 'lat',
        'longitude': 'lon'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    numeric_cols = [
        'year','electricity_access','renewable_share','co2_kt','energy_per_capita',
        'energy_intensity','gdp_pc','gdp_growth','climate_finance','elec_fossil_twh',
        'elec_ren_twh','low_carbon_pct','lat','lon'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def pct_change(a, b):
    if a is None or np.isnan(a) or a == 0:
        return np.nan
    return (b - a) / a * 100.0

def build_first_last_summary(df, keys, label):
    g = (df.groupby(keys + ['year'], as_index=False)
           .agg({'electricity_access':'mean','renewable_share':'mean','co2_kt':'sum'}))
    out = []
    for key_vals, sub in g.groupby(keys):
        sub = sub.sort_values('year')
        y0, y1 = sub['year'].iloc[0], sub['year'].iloc[-1]
        r0, r1 = sub.iloc[0], sub.iloc[-1]
        row = {}
        if isinstance(key_vals, tuple):
            for k, v in zip(keys, key_vals):
                row[k] = v
        else:
            row[keys[0]] = key_vals
        row.update({
            'year_start': int(y0), 'year_end': int(y1),
            'access_start': r0.electricity_access, 'access_end': r1.electricity_access,
            'access_pct_change': pct_change(r0.electricity_access, r1.electricity_access),
            're_start': r0.renewable_share, 're_end': r1.renewable_share,
            're_pct_change': pct_change(r0.renewable_share, r1.renewable_share),
            'co2_start_kt': r0.co2_kt, 'co2_end_kt': r1.co2_kt,
            'co2_pct_change': pct_change(r0.co2_kt, r1.co2_kt),
            'category': label
        })
        out.append(row)
    return pd.DataFrame(out)

def run_ols(y, X, add_const=True):
    import statsmodels.api as sm
    if add_const:
        X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X, missing='drop')
    return model.fit()

# ---------------------------
# Data (local only)
# ---------------------------
DATA_PATH = "cleaned_dataset.csv"

try:
    df_raw = load_data_from_path(DATA_PATH)
except Exception as e:
    st.error(f"Could not read '{DATA_PATH}'. Make sure it is in the same folder as this app. Error: {e}")
    st.stop()

df = rename_and_prepare(df_raw)

# ---------------------------
# Sidebar filters (with Exclude 2020)
# ---------------------------
with st.sidebar:
    st.header("Controls")

    # Exclude 2020 toggle (ON by default)
    exclude_2020 = st.toggle("Exclude 2020 (partial pandemic year)", value=True)
    if exclude_2020 and 'year' in df.columns and 2020 in df['year'].unique():
        df = df[df['year'] != 2020]

    min_year, max_year = int(df['year'].min()), int(df['year'].max())
    year_range = st.slider("Year range", min_year, max_year, (min_year, max_year), step=1)

    regions = sorted(df['region'].dropna().unique()) if 'region' in df.columns else []
    incomes = sorted(df['income'].dropna().unique()) if 'income' in df.columns else []
    countries = sorted(df['country'].dropna().unique()) if 'country' in df.columns else []

    sel_regions = st.multiselect("Regions", regions, default=regions)
    sel_incomes = st.multiselect("Income groups", incomes, default=incomes)
    sel_countries = st.multiselect("Countries (optional filter)", [])

    researcher_mode = st.toggle("Researcher Mode (advanced)", value=False)

# apply filters
f = df['year'].between(year_range[0], year_range[1])
if sel_regions and 'region' in df.columns: f &= df['region'].isin(sel_regions)
if sel_incomes and 'income' in df.columns: f &= df['income'].isin(sel_incomes)
if sel_countries and 'country' in df.columns: f &= df['country'].isin(sel_countries)
dff = df[f].copy()

# ---------------------------
# HERO (text only)
# ---------------------------
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown("### 🌍 Global Energy, Renewables & SDGs")
st.markdown("""
This dashboard shows three simple ideas:
1) **Access** — how many people have electricity.  
2) **Renewables** — how much of energy use is clean.  
3) **CO₂** — total emissions.  
Use the filters on the left to focus on years, regions, or income groups.
""")
st.markdown("""
<span class="help-chip">Electricity Access = % of people with power</span>
<span class="help-chip">Renewable Share = % of total energy that’s renewable</span>
<span class="help-chip">CO₂ (kt) = Thousand tonnes of carbon dioxide</span>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Tabs
# ---------------------------
tabs_public = [
    "Overview", "Developed vs Developing", "Rebound Patterns",
    "SDG Alignment", "Country Explorer", "For Everyone (Explaination)"
]
tabs_all = tabs_public.copy()
if researcher_mode:
    tabs_all.insert(5, "Economic Moderators")
tabs_all.append("Downloads")
tabs = st.tabs(tabs_all)

# ---------------------------
# Overview
# ---------------------------
with tabs[tabs_all.index("Overview")]:
    st.markdown("## Global Overview")
    st.caption("At a glance: average access and renewables, and global total CO₂.")

    need = {'year','electricity_access','renewable_share','co2_kt'}
    if need.issubset(dff.columns):
        g = (dff.groupby('year', as_index=False)
               .agg({'electricity_access':'mean','renewable_share':'mean','co2_kt':'sum'})
               .sort_values('year'))

        y0, y1 = g['year'].iloc[0], g['year'].iloc[-1]
        r0, r1 = g.iloc[0], g.iloc[-1]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("Electricity Access (avg %)", f"{r1.electricity_access:,.2f}",
                      f"{pct_change(r0.electricity_access, r1.electricity_access):.1f}% vs {y0}")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("Renewable Share (avg %)", f"{r1.renewable_share:,.2f}",
                      f"{pct_change(r0.renewable_share, r1.renewable_share):.1f}% vs {y0}")
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("CO₂ (kt, total)", f"{r1.co2_kt:,.0f}",
                      f"{pct_change(r0.co2_kt, r1.co2_kt):.1f}% vs {y0}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.plotly_chart(
            px.line(g, x="year", y=["electricity_access","renewable_share"],
                    title="Global: Electricity Access & Renewable Share",
                    labels={"value":"%","variable":""}),
            use_container_width=True
        )
        st.caption("**Interpretation:** upward lines mean progress; if both rise together, more people get power and a larger share is clean.")

        st.plotly_chart(px.line(g, x="year", y="co2_kt", title="Global: CO₂ Emissions (kt)"),
                        use_container_width=True)
        st.caption("**Note:** CO₂ can still rise even when renewables grow, if total demand grows faster.")
    else:
        st.warning("Missing columns for overview.")

# ---------------------------
# Developed vs Developing
# ---------------------------
with tabs[tabs_all.index("Developed vs Developing")]:
    st.markdown("## Compare by Income Group and Region")
    st.caption("See how richer and poorer economies differ in access, renewables, and emissions.")

    if {'income','year'}.issubset(dff.columns):
        gi = (dff.groupby(['income','year'], as_index=False)
                .agg({'renewable_share':'mean','co2_kt':'sum','electricity_access':'mean'})
                .sort_values(['income','year']))
        st.plotly_chart(
            px.line(gi, x='year', y='renewable_share', color='income', markers=True,
                    title="Renewable Share (%) by Income Group"),
            use_container_width=True
        )
        st.caption("**Takeaway:** high income often rises slowly; lower-middle income may change faster but from a lower base.")

        st.plotly_chart(
            px.line(gi, x='year', y='electricity_access', color='income', markers=True,
                    title="Electricity Access (%) by Income Group"),
            use_container_width=True
        )
        st.caption("**Access gap:** many low-income countries still catch up toward near-universal access.")

    # Animated map only if lat/lon available
    if {'country','renewable_share','lat','lon','year'}.issubset(dff.columns):
        sub = dff.dropna(subset=['lat','lon','renewable_share'])
        if not sub.empty:
            figm = px.scatter_geo(
                sub, lat='lat', lon='lon', hover_name='country',
                color='renewable_share', color_continuous_scale='Viridis',
                projection='natural earth', animation_frame='year',
                title="Animated Map – Renewable Share (%) by Year"
            )
            st.plotly_chart(figm, use_container_width=True)
            st.caption("**How to read:** darker color = higher renewable share. Use the play bar to see change over time.")

    with st.expander("India & China (quick look)"):
        ic = dff[dff.get('country','').isin(['India','China'])].groupby(['country','year'], as_index=False).agg(
            co2_kt=('co2_kt','sum'),
            renewable_share=('renewable_share','mean')
        )
        if not ic.empty:
            cols = st.columns(2)
            cols[0].plotly_chart(px.line(ic, x='year', y='co2_kt', color='country',
                                         title="CO₂ – India vs China"), use_container_width=True)
            cols[1].plotly_chart(px.line(ic, x='year', y='renewable_share', color='country',
                                         title="Renewables – India vs China"), use_container_width=True)
            st.caption("**Context:** both added renewables, but CO₂ trends reflect industrial growth and rising demand (‘energy stacking’).")

# ---------------------------
# Rebound Patterns
# ---------------------------
with tabs[tabs_all.index("Rebound Patterns")]:
    st.markdown("## Rebound & Demand Patterns")
    st.caption("If energy becomes more available or cheaper, people may use more. These simple proxies show that possibility.")

    need1 = {'year','electricity_access','renewable_share'}
    if need1.issubset(dff.columns):
        gg = dff.groupby('year', as_index=False)[['electricity_access','renewable_share']].mean()
        fig1 = go.Figure()
        fig1.add_scatter(x=gg['year'], y=gg['electricity_access'], name='Electricity Access (%)')
        fig1.add_scatter(x=gg['year'], y=gg['renewable_share'], name='Renewable Share (%)')
        fig1.update_layout(title="Household Proxy: Access vs Renewables", xaxis_title="Year", yaxis_title="%")
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("**Reading:** if both rise together, more households get electricity while renewables expand.")

    need2 = {'year','energy_intensity','renewable_share'}
    if need2.issubset(dff.columns):
        gg2 = dff.groupby('year', as_index=False)[['energy_intensity','renewable_share']].mean()
        fig2 = go.Figure()
        fig2.add_scatter(x=gg2['year'], y=gg2['energy_intensity'], name='Energy Intensity (MJ/$2017 PPP GDP)')
        fig2.add_scatter(x=gg2['year'], y=gg2['renewable_share'], name='Renewable Share (%)', yaxis='y2')
        fig2.update_layout(
            title="Transport/Efficiency Proxy", xaxis_title="Year",
            yaxis=dict(title="Energy Intensity"),
            yaxis2=dict(title="Renewable Share (%)", overlaying='y', side='right')
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("**Idea:** falling intensity with rising renewables suggests cleaner and more efficient economies.")

    need3 = {'year','energy_per_capita','renewable_share'}
    if need3.issubset(dff.columns):
        gg3 = dff.groupby('year', as_index=False)[['energy_per_capita','renewable_share']].mean()
        fig3 = go.Figure()
        fig3.add_scatter(x=gg3['year'], y=gg3['energy_per_capita'], name='Energy per Capita (kWh/person)')
        fig3.add_scatter(x=gg3['year'], y=gg3['renewable_share'], name='Renewable Share (%)', yaxis='y2')
        fig3.update_layout(
            title="Industry/Overall Proxy", xaxis_title="Year",
            yaxis=dict(title="Energy per Capita"),
            yaxis2=dict(title="Renewable Share (%)", overlaying='y', side='right')
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("**Caution:** energy per person can keep rising even with more renewables — a classic rebound pattern.")

    with st.expander("Show top countries where energy per person rises with renewables (diagnostic)"):
        need4 = {'country','renewable_share','energy_per_capita','year'}
        if need4.issubset(dff.columns):
            dfp = dff[list(need4)].dropna().copy()
            slopes = []
            for c, sub in dfp.groupby('country'):
                sub = sub.sort_values('year')
                if sub['renewable_share'].nunique() >= 2 and len(sub) >= 3:
                    slope = np.polyfit(sub['renewable_share'], sub['energy_per_capita'], 1)[0]
                    slopes.append({'country': c, 'slope_Epc_per_1pctRE': slope})
            slopes_df = pd.DataFrame(slopes)
            if not slopes_df.empty:
                st.dataframe(slopes_df.sort_values('slope_Epc_per_1pctRE', ascending=False).head(10), use_container_width=True)
                st.caption("Higher slope suggests energy-per-person tends to rise alongside renewables (possible rebound/stacking).")

# ---------------------------
# SDG Alignment
# ---------------------------
with tabs[tabs_all.index("SDG Alignment")]:
    st.markdown("## SDG Alignment & Snapshots")
    st.caption("SDG 7: Access & Renewables | SDG 13: Climate action (CO₂ as a simple proxy).")

    need = {'year','electricity_access','renewable_share','co2_kt'}
    if need.issubset(dff.columns):
        global_ts = (dff.groupby('year', as_index=False)
                       .agg({'electricity_access':'mean','renewable_share':'mean','co2_kt':'sum'})
                       .sort_values('year'))
        base_year = int(global_ts['year'].min())
        base = global_ts.loc[global_ts['year']==base_year].iloc[0]
        idx = global_ts.copy()
        idx['elec_idx'] = (idx['electricity_access']/base.electricity_access)*100
        idx['re_idx']   = (idx['renewable_share']/base.renewable_share)*100
        idx['co2_idx']  = (idx['co2_kt']/base.co2_kt)*100

        fig_idx = go.Figure()
        fig_idx.add_scatter(x=idx['year'], y=idx['elec_idx'], name='Access (index)')
        fig_idx.add_scatter(x=idx['year'], y=idx['re_idx'], name='RE share (index)')
        fig_idx.add_scatter(x=idx['year'], y=idx['co2_idx'], name='CO₂ (index)')
        fig_idx.update_layout(title=f"Indexed SDG Indicators (base={base_year})",
                              xaxis_title="Year", yaxis_title="Index (base=100)")
        st.plotly_chart(fig_idx, use_container_width=True)
        st.caption("**Indexed view:** all series start at 100 in the first year, so growth is easy to compare.")

        cols = ['country','region','income','year','electricity_access','renewable_share','co2_kt']
        sub = dff[[c for c in cols if c in dff.columns]].copy()

        if 'region' in sub.columns:
            reg_sum = build_first_last_summary(sub, ['region'], 'Region').sort_values('region')
            st.markdown("**2000 vs last year – by Region**")
            st.dataframe(reg_sum, use_container_width=True)

        if 'income' in sub.columns:
            inc_sum = build_first_last_summary(sub, ['income'], 'Income').sort_values('income')
            st.markdown("**2000 vs last year – by Income Group**")
            st.dataframe(inc_sum, use_container_width=True)

        with st.expander("Top movers since base year (countries)"):
            base_y = int(dff['year'].min())
            last_y = int(dff['year'].max())
            base_cty = dff[dff['year']==base_y][['country','renewable_share','co2_kt']].rename(
                columns={'renewable_share':'re0','co2_kt':'co20'})
            last_cty = dff[dff['year']==last_y][['country','renewable_share','co2_kt']].rename(
                columns={'renewable_share':'re1','co2_kt':'co21'})
            ch = base_cty.merge(last_cty, on='country', how='inner').dropna()
            if not ch.empty:
                ch['re_pct_change']  = (ch['re1'] - ch['re0']) / ch['re0'] * 100
                ch['co2_pct_change'] = (ch['co21'] - ch['co20']) / ch['co20'] * 100
                c1, c2 = st.columns(2)
                c1.markdown("**Top 5 ↑ Renewable Share (% change)**")
                c1.dataframe(ch.sort_values('re_pct_change', ascending=False).head(5)[['country','re_pct_change']])
                c2.markdown("**Top 5 ↓ CO₂ (% change)**")
                c2.dataframe(ch.sort_values('co2_pct_change').head(5)[['country','co2_pct_change']])
        st.info("**Use cases:** quick policy briefs, presentations, and regional comparisons.")
    else:
        st.info("Missing columns for SDG analysis.")

# ---------------------------
# Economic Moderators (Advanced – Researcher Mode)
# ---------------------------
if researcher_mode:
    with tabs[tabs_all.index("Economic Moderators")]:
        st.markdown("## Economic Moderators (Advanced)")
        st.caption("For expert readers. OLS tables and an EKC-shaped relationship if present.")
        mdf = dff.dropna(subset=['co2_kt','renewable_share']).copy()

        if {'co2_kt','renewable_share'}.issubset(mdf.columns):
            try:
                res1 = run_ols(mdf['co2_kt'], mdf[['renewable_share']])
                st.text(res1.summary())
            except Exception as e:
                st.warning(f"Baseline OLS failed: {e}")

        need = {'co2_kt','renewable_share','gdp_pc','energy_intensity'}
        if need.issubset(mdf.columns):
            mdf2 = mdf.dropna(subset=['gdp_pc','energy_intensity']).copy()
            if len(mdf2) > 30:
                try:
                    res2 = run_ols(mdf2['co2_kt'], mdf2[['renewable_share','gdp_pc','energy_intensity']])
                    st.text(res2.summary())
                except Exception as e:
                    st.warning(f"Expanded OLS failed: {e}")

        if {'co2_kt','gdp_pc','renewable_share'}.issubset(mdf.columns):
            mdf3 = mdf.dropna(subset=['gdp_pc']).copy()
            if len(mdf3) > 30:
                mdf3['gdp_pc_sq'] = mdf3['gdp_pc']**2
                try:
                    res3 = run_ols(mdf3['co2_kt'], mdf3[['gdp_pc','gdp_pc_sq','renewable_share']])
                    st.text(res3.summary())
                    mdf3b = mdf3.copy()
                    mdf3b['gdp_bin'] = pd.qcut(mdf3b['gdp_pc'], q=20, duplicates='drop')
                    ekc = mdf3b.groupby('gdp_bin', as_index=False)['co2_kt'].mean()
                    ekc['gdp_mid'] = ekc['gdp_bin'].apply(lambda x: x.mid)
                    st.plotly_chart(px.line(ekc, x='gdp_mid', y='co2_kt', markers=True,
                                            title="EKC-style curve (binned GDP per capita)"),
                                    use_container_width=True)
                except Exception as e:
                    st.warning(f"EKC model failed: {e}")

# ---------------------------
# Country Explorer
# ---------------------------
with tabs[tabs_all.index("Country Explorer")]:
    st.markdown("## Country Explorer")
    st.caption("Pick a country and see the trend lines. Higher = more of that measure.")

    if 'country' in dff.columns:
        ctry = st.selectbox("Select a country", options=sorted(dff['country'].dropna().unique()))
        sub = dff[dff['country'] == ctry].sort_values('year')
        if sub.empty:
            st.info("No data for this selection.")
        else:
            c1, c2 = st.columns(2)
            if 'co2_kt' in sub.columns and sub['co2_kt'].notna().any():
                c1.plotly_chart(px.line(sub, x='year', y='co2_kt',
                                        title=f"{ctry} – CO₂ Emissions (kt)"), use_container_width=True)
            if 'renewable_share' in sub.columns and sub['renewable_share'].notna().any():
                c2.plotly_chart(px.line(sub, x='year', y='renewable_share',
                                        title=f"{ctry} – Renewable Share (%)"), use_container_width=True)
            if 'electricity_access' in sub.columns and sub['electricity_access'].notna().any():
                st.plotly_chart(px.line(sub, x='year', y='electricity_access',
                                        title=f"{ctry} – Electricity Access (%)"), use_container_width=True)
            st.caption("**Tip:** if renewables go up but CO₂ doesn’t drop, demand might be rising at the same time (energy stacking).")

# ---------------------------
# For Everyone (Explain)
# ---------------------------
with tabs[tabs_all.index("For Everyone (Explaination)")]:
    st.markdown("## Hi,  I’m New to Energy Data")
    st.markdown("""
**Electricity Access** — % of people with electricity at home (higher is better).  
**Renewable Share** — % of total energy that comes from renewables (higher is better).  
**CO₂ (kt)** — Total carbon dioxide emissions in thousand tonnes (lower is better).  

**Why can renewables ↑ while CO₂ doesn’t ↓?**  
Demand can grow fast. A country may add clean energy **and** still use more fossil power — that’s energy stacking.

**Reading the charts**  
- Lines ↑ mean increase; ↓ mean decrease.  
- Indexed chart sets the first year to 100 to compare growth fairly.  
- Some charts have two y-axes (left/right). The legend shows which line uses which axis.
""")

# ---------------------------
# Downloads
# ---------------------------
with tabs[tabs_all.index("Downloads")]:
    st.markdown("## Downloads & Tables")
    st.caption("Download the exact slice you’re viewing and quick summary tables for reports/policy briefs.")

    keep_cols = ['country','region','income','year','electricity_access','renewable_share','co2_kt',
                 'energy_per_capita','energy_intensity','gdp_pc','gdp_growth','climate_finance']
    pub = dff[[c for c in keep_cols if c in dff.columns]].dropna(how='all')
    st.markdown("**Filtered dataset (based on current filters)**")
    st.dataframe(pub.head(50), use_container_width=True)
    st.download_button("Download CSV (current filters)",
                       pub.to_csv(index=False), file_name="public_energy_sdg_filtered.csv", mime="text/csv")

    cols = ['country','region','income','year','electricity_access','renewable_share','co2_kt']
    sub = dff[[c for c in cols if c in dff.columns]].copy()
    gsum = build_first_last_summary(sub.assign(group='Global'), ['group'], 'Global') if not sub.empty else pd.DataFrame()
    rsum = build_first_last_summary(sub, ['region'], 'Region') if 'region' in sub.columns else pd.DataFrame()
    isum = build_first_last_summary(sub, ['income'], 'Income') if 'income' in sub.columns else pd.DataFrame()
    summary = pd.concat([gsum, rsum, isum], ignore_index=True)

    st.markdown("**2000 vs last year summaries (Global / Region / Income)**")
    st.dataframe(summary, use_container_width=True)
    st.download_button("Download SDG summary CSV",
                       summary.to_csv(index=False), file_name="sdg_2000_vs_last_summary.csv", mime="text/csv")
