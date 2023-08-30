import streamlit as st
import altair as alt

from collections import OrderedDict

from datetime import date, timedelta
import numpy as np
import pandas as pd

import utils  # streamlit runs from root directory, so we can import utils directly

def generate_plots(minimum_m_df):
    c = alt.Chart(minimum_m_df).mark_line().encode(
        x='exchange_rate:Q',
        y='minimum_m:Q',
        color='onboarding_scenario:N',
    )
    st.altair_chart(c, use_container_width=True)

def compute_minimum_multiplier(scenario2erpt=None):
    borrowing_cost_pct = st.session_state['mm_borrow_cost_pct'] / 100.0
    filp_bd_cost_tib_per_yr = st.session_state['mm_filp_bizdev_cost']
    deal_income_tib_per_yr = st.session_state['mm_deal_income']
    data_prep_cost_tib_per_yr = st.session_state['mm_data_prep_cost']
    penalty_tib_per_yr = st.session_state['mm_cheating_penalty']

    power_cost_tib_per_yr = st.session_state['mm_power_cost']
    bw_cost_tib_per_yr = st.session_state['mm_bw_cost']
    staff_cost_tib_per_yr = st.session_state['mm_staff_cost']

    sealing_costs_tib_per_yr, gas_cost_tib_per_yr, _, _ = utils.get_negligible_costs(bw_cost_tib_per_yr)

    # sweep across a) exchange rate, b) onboarding scenario
    exchange_rate_vec = np.linspace(3, 20.0, 100)
    onboarding_scenario_vec = ['status-quo', 'pessimistic', 'optimistic']

    minimum_m_results = []
    for onboarding_scenario in onboarding_scenario_vec:
        erpt = scenario2erpt[onboarding_scenario]
        for exchange_rate in exchange_rate_vec:
            sector_return_nomult = erpt*exchange_rate
            revenue = deal_income_tib_per_yr

            cost_multiplier = sector_return_nomult*borrowing_cost_pct
            cost_no_multiplier = (
                gas_cost_tib_per_yr +
                power_cost_tib_per_yr +
                bw_cost_tib_per_yr +
                staff_cost_tib_per_yr +
                sealing_costs_tib_per_yr +
                data_prep_cost_tib_per_yr +
                filp_bd_cost_tib_per_yr +
                (staff_cost_tib_per_yr+power_cost_tib_per_yr)*0.5 +
                penalty_tib_per_yr
            )

            minimum_m = (cost_no_multiplier - revenue + sector_return_nomult)/(sector_return_nomult - cost_multiplier)
            minimum_m_results.append({
                'onboarding_scenario': onboarding_scenario,
                'exchange_rate': exchange_rate,
                'minimum_m': minimum_m
            })
    minimum_m_df = pd.DataFrame(minimum_m_results)
    generate_plots(minimum_m_df)
        

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
    st.selectbox(
        'Onboarding Scenario', ('Status-Quo', 'Pessimistic', 'Optimistic'), key="mm_onboarding_scenario",
        on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
    )
    with st.expander("Revenue Settings", expanded=False):
        st.slider(
            'Deal Income ($/TiB/Yr)', 
            min_value=0.0, max_value=100.0, value=16.0, step=1.0, format='%0.02f', key="mm_deal_income",
            on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
    with st.expander("Cost Settings", expanded=False):
        st.slider(
            'Borrowing Costs (Pct. of Pledge)', 
            min_value=0.0, max_value=100.0, value=50.0, step=1.00, format='%0.02f', key="mm_borrow_cost_pct",
            on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'FIL+ Biz Dev Cost ($/TiB/Yr)', 
            min_value=1.0, max_value=50.0, value=8.0, step=1.0, format='%0.02f', key="mm_filp_bizdev_cost",
            on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'RD Biz Dev Cost ($/TiB/Yr)', 
            min_value=1.0, max_value=50.0, value=3.2, step=1.0, format='%0.02f', key="mm_rd_bizdev_cost",
            on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Data Prep Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=1.0, step=1.0, format='%0.02f', key="mm_data_prep_cost",
            on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Cheating Penalty ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=18.0, step=1.0, format='%0.02f', key="mm_cheating_penalty",
            on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Power+COLO Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=6.0, step=1.0, format='%0.02f', key="mm_power_cost",
            on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Bandwidth [10GBPS] Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=6.0, step=1.0, format='%0.02f', key="mm_bw_cost",
            on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Staff Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=8.0, step=1.0, format='%0.02f', key="mm_staff_cost",
            on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
    st.button("Compute!", on_click=compute_minimum_multiplier, kwargs=kwargs, key="forecast_button")