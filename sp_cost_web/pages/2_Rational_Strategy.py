import streamlit as st
import altair as alt

from collections import OrderedDict

from datetime import date, timedelta
import numpy as np
import pandas as pd

import utils  # streamlit runs from root directory, so we can import utils directly

st.set_page_config(
    page_title="Parametric Exploration of Rational Strategy", 
    page_icon=":brain:",
    layout="wide",
)

def generate_plots(borrowing_cost_df, deal_income_plot_df, data_prepcost_plot_df, bizdev_cost_plot_df):
    st.write("### Exploration of rational strategy conditioned on individual factors")

    col1, col2 = st.columns(2)

    with col1:
        borrowing_cost_chart = alt.Chart(borrowing_cost_df, title="Borrowing Cost").mark_line().encode(
            x=alt.X('borrowing_cost_pct:Q').title('Borrowing Cost Pct [%]'),
            y=alt.Y('profit:Q').title("Net Income [$/TiB/Yr]"),
            color=alt.Color('SP Type:O', scale=alt.Scale(scheme='tableau20')),
                tooltip=[
                alt.Tooltip('SP Type', title='Strategy'),
                alt.Tooltip('profit', title='Profit'),
                alt.Tooltip('borrowing_cost_pct', title='Borrowing Cost [\% of Block Rewards]', format='.2f'),
            ]
        )
        st.altair_chart(borrowing_cost_chart, use_container_width=True)

        data_prepcost_chart = alt.Chart(data_prepcost_plot_df, title="Data-Prep Cost").mark_line().encode(
            x=alt.X('data_prepcost:Q').title('Data-Prep Cost [$/TiB/Yr]'),
            y=alt.Y('profit:Q').title("Net Income [$/TiB/Yr]"),
            color=alt.Color('SP Type:O', scale=alt.Scale(scheme='tableau20')),
                tooltip=[
                alt.Tooltip('SP Type', title='Strategy'),
                alt.Tooltip('profit', title='Profit'),
                alt.Tooltip('data_prepcost', title='Data-Prep Cost', format='.2f'),
            ]
        )
        st.altair_chart(data_prepcost_chart, use_container_width=True)

    with col2:
        deal_income_chart = alt.Chart(deal_income_plot_df, title="Deal Income").mark_line().encode(
            x=alt.X('deal_income:Q').title('Deal Income [$/TiB/Yr]'),
            y=alt.Y('profit:Q').title("Net Income [$/TiB/Yr]"),
            color=alt.Color('SP Type:O', scale=alt.Scale(scheme='tableau20')),
                tooltip=[
                alt.Tooltip('SP Type', title='Strategy'),
                alt.Tooltip('profit', title='Profit'),
                alt.Tooltip('deal_income', title='Deal Income', format='.2f'),
            ]
        )
        st.altair_chart(deal_income_chart, use_container_width=True)

        bizdev_cost_chart = alt.Chart(bizdev_cost_plot_df, title="BizDev Cost [Assumption: FIL+=2x RD Cost]").mark_line().encode(
            x=alt.X('bizdev_cost:Q').title('BizDev Cost [$/TiB/Yr]'),
            y=alt.Y('profit:Q').title("Net Income [$/TiB/Yr]"),
            color=alt.Color('SP Type:O', scale=alt.Scale(scheme='tableau20')),
                tooltip=[
                alt.Tooltip('SP Type', title='Strategy'),
                alt.Tooltip('profit', title='Profit'),
                alt.Tooltip('bizdev_cost', title='BizDev Cost', format='.2f'),
            ]
        )
        st.altair_chart(bizdev_cost_chart, use_container_width=True)



def generate_rankings(scenario2erpt=None):
    # get fixed costs
    filp_multiplier = st.session_state['rs_filp_multiplier']
    rd_multiplier = st.session_state['rs_rd_multiplier']
    cc_multiplier = st.session_state['rs_cc_multiplier']

    onboarding_scenario = st.session_state['rs_onboarding_scenario'].lower()
    
    exchange_rate =  st.session_state['rs_filprice_slider']
    borrowing_cost_pct = st.session_state['rs_borrow_cost_pct'] / 100.0
    filp_bd_cost_tib_per_yr = st.session_state['rs_filp_bizdev_cost']
    rd_bd_cost_tib_per_yr = st.session_state['rs_rd_bizdev_cost']
    deal_income_tib_per_yr = st.session_state['rs_deal_income']
    data_prep_cost_tib_per_yr = st.session_state['rs_data_prep_cost']
    penalty_tib_per_yr = st.session_state['rs_cheating_penalty']

    power_cost_tib_per_yr = st.session_state['rs_power_cost']
    bw_cost_tib_per_yr = st.session_state['rs_bw_cost']
    staff_cost_tib_per_yr = st.session_state['rs_staff_cost']

    # sweep borrowing_cost, fix other costs
    borrowing_cost_vec = np.linspace(0,100,25)
    borrowing_cost_plot_vec = []
    for borrowing_cost_sweep_pct in borrowing_cost_vec:
        borrowing_cost_sweep_frac = borrowing_cost_sweep_pct/100.0
        df = utils.compute_costs(
            scenario2erpt=scenario2erpt,
            filp_multiplier=filp_multiplier, rd_multiplier=rd_multiplier, cc_multiplier=cc_multiplier,
            onboarding_scenario=onboarding_scenario,
            exchange_rate=exchange_rate, borrowing_cost_pct=borrowing_cost_sweep_frac,
            filp_bd_cost_tib_per_yr=filp_bd_cost_tib_per_yr, rd_bd_cost_tib_per_yr=rd_bd_cost_tib_per_yr,
            deal_income_tib_per_yr=deal_income_tib_per_yr,
            data_prep_cost_tib_per_yr=data_prep_cost_tib_per_yr, penalty_tib_per_yr=penalty_tib_per_yr,
            power_cost_tib_per_yr=power_cost_tib_per_yr, bandwidth_10gbps_tib_per_yr=bw_cost_tib_per_yr,
            staff_cost_tib_per_yr=staff_cost_tib_per_yr
        )
        df['borrowing_cost_pct'] = borrowing_cost_sweep_pct
        df['rank'] = df.sort_values(by='profit', ascending=False).index.values        
        borrowing_cost_plot_vec.append(df[['SP Type', 'rank', 'borrowing_cost_pct', 'profit']])
    borrowing_cost_plot_df = pd.concat(borrowing_cost_plot_vec)

    # sweep deal_income
    deal_income_vec = np.linspace(0,100,25)
    deal_income_plot_vec = []
    for deal_income_sweep in deal_income_vec:
        df = utils.compute_costs(
            scenario2erpt=scenario2erpt,
            filp_multiplier=filp_multiplier, rd_multiplier=rd_multiplier, cc_multiplier=cc_multiplier,
            onboarding_scenario=onboarding_scenario,
            exchange_rate=exchange_rate, borrowing_cost_pct=borrowing_cost_pct,
            filp_bd_cost_tib_per_yr=filp_bd_cost_tib_per_yr, rd_bd_cost_tib_per_yr=rd_bd_cost_tib_per_yr,
            deal_income_tib_per_yr=deal_income_sweep,
            data_prep_cost_tib_per_yr=data_prep_cost_tib_per_yr, penalty_tib_per_yr=penalty_tib_per_yr,
            power_cost_tib_per_yr=power_cost_tib_per_yr, bandwidth_10gbps_tib_per_yr=bw_cost_tib_per_yr,
            staff_cost_tib_per_yr=staff_cost_tib_per_yr
        )
        df['deal_income'] = deal_income_sweep
        df['rank'] = df.sort_values(by='profit', ascending=False).index.values        
        deal_income_plot_vec.append(df[['SP Type', 'rank', 'deal_income', 'profit']])
    deal_income_plot_df = pd.concat(deal_income_plot_vec)

    # sweep data_prep_cost
    data_prepcost_vec = np.linspace(0,100,25)
    data_prepcost_plot_vec = []
    for data_prepcost_sweep in data_prepcost_vec:
        df = utils.compute_costs(
            scenario2erpt=scenario2erpt,
            filp_multiplier=filp_multiplier, rd_multiplier=rd_multiplier, cc_multiplier=cc_multiplier,
            onboarding_scenario=onboarding_scenario,
            exchange_rate=exchange_rate, borrowing_cost_pct=borrowing_cost_pct,
            filp_bd_cost_tib_per_yr=filp_bd_cost_tib_per_yr, rd_bd_cost_tib_per_yr=rd_bd_cost_tib_per_yr,
            deal_income_tib_per_yr=deal_income_tib_per_yr,
            data_prep_cost_tib_per_yr=data_prepcost_sweep, penalty_tib_per_yr=penalty_tib_per_yr,
            power_cost_tib_per_yr=power_cost_tib_per_yr, bandwidth_10gbps_tib_per_yr=bw_cost_tib_per_yr,
            staff_cost_tib_per_yr=staff_cost_tib_per_yr
        )
        df['data_prepcost'] = data_prepcost_sweep
        df['rank'] = df.sort_values(by='profit', ascending=False).index.values        
        data_prepcost_plot_vec.append(df[['SP Type', 'rank', 'data_prepcost', 'profit']])
    data_prepcost_plot_df = pd.concat(data_prepcost_plot_vec)

    # sweep bizdev_cost
    #  assume RD bizdev cost = 50% of FIL+ bizdev cost
    bizdev_cost_vec = np.linspace(0,100,25)
    bizdev_cost_plot_vec = []
    for bizdev_cost in bizdev_cost_vec:
        filp_bizdev_cost = bizdev_cost
        rd_bizdev_cost = bizdev_cost * 0.5
        df = utils.compute_costs(
            scenario2erpt=scenario2erpt,
            filp_multiplier=filp_multiplier, rd_multiplier=rd_multiplier, cc_multiplier=cc_multiplier,
            onboarding_scenario=onboarding_scenario,
            exchange_rate=exchange_rate, borrowing_cost_pct=borrowing_cost_pct,
            filp_bd_cost_tib_per_yr=filp_bizdev_cost, rd_bd_cost_tib_per_yr=rd_bizdev_cost,
            deal_income_tib_per_yr=deal_income_tib_per_yr,
            data_prep_cost_tib_per_yr=data_prep_cost_tib_per_yr, penalty_tib_per_yr=penalty_tib_per_yr,
            power_cost_tib_per_yr=power_cost_tib_per_yr, bandwidth_10gbps_tib_per_yr=bw_cost_tib_per_yr,
            staff_cost_tib_per_yr=staff_cost_tib_per_yr
        )
        df['bizdev_cost'] = bizdev_cost
        df['rank'] = df.sort_values(by='profit', ascending=False).index.values        
        bizdev_cost_plot_vec.append(df[['SP Type', 'rank', 'bizdev_cost', 'profit']])
    bizdev_cost_plot_df = pd.concat(bizdev_cost_plot_vec)

    # plot
    generate_plots(borrowing_cost_plot_df, deal_income_plot_df, data_prepcost_plot_df, bizdev_cost_plot_df)

current_date = date.today() - timedelta(days=3)
mo_start = min(current_date.month - 1 % 12, 1)
start_date = date(current_date.year, mo_start, 1)
forecast_length_days=365*3
end_date = current_date + timedelta(days=forecast_length_days)
scenario2erpt = utils.get_offline_data(start_date, current_date, end_date)  # should be cached
kwargs = {
    'scenario2erpt':scenario2erpt
}

with st.sidebar:
    st.slider(
        "FIL Exchange Rate ($/FIL)", 
        min_value=3., max_value=50., value=4.0, step=.1, format='%0.02f', key="rs_filprice_slider",
        on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
    )
    st.selectbox(
        'Onboarding Scenario', ('Status-Quo', 'Pessimistic', 'Optimistic'), key="rs_onboarding_scenario",
        on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
    )
    with st.expander("Revenue Settings", expanded=False):
        st.slider(
            'Deal Income ($/TiB/Yr)', 
            min_value=0.0, max_value=100.0, value=16.0, step=1.0, format='%0.02f', key="rs_deal_income",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
    with st.expander("Cost Settings", expanded=False):
        st.slider(
            'Borrowing Costs (Pct. of Pledge)', 
            min_value=0.0, max_value=100.0, value=50.0, step=1.00, format='%0.02f', key="rs_borrow_cost_pct",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'FIL+ Biz Dev Cost ($/TiB/Yr)', 
            min_value=1.0, max_value=50.0, value=8.0, step=1.0, format='%0.02f', key="rs_filp_bizdev_cost",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'RD Biz Dev Cost ($/TiB/Yr)', 
            min_value=1.0, max_value=50.0, value=3.2, step=1.0, format='%0.02f', key="rs_rd_bizdev_cost",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Data Prep Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=1.0, step=1.0, format='%0.02f', key="rs_data_prep_cost",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Cheating Penalty ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=0.0, step=1.0, format='%0.02f', key="rs_cheating_penalty",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Power+COLO Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=6.0, step=1.0, format='%0.02f', key="rs_power_cost",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Bandwidth [10GBPS] Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=6.0, step=1.0, format='%0.02f', key="rs_bw_cost",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Staff Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=10.0, value=6.0, step=1.0, format='%0.02f', key="rs_staff_cost",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
    with st.expander("Multipliers", expanded=False):
        st.slider(
            'CC', min_value=1, max_value=20, value=1, step=1, key="rs_cc_multiplier",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'RD', min_value=1, max_value=20, value=1, step=1, key="rs_rd_multiplier",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'FIL+', min_value=1, max_value=20, value=10, step=1, key="rs_filp_multiplier",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
    st.button("Compute!", on_click=generate_rankings, kwargs=kwargs, key="forecast_button")