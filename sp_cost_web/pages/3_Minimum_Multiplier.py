import streamlit as st
import altair as alt

from collections import OrderedDict

from datetime import date, timedelta
import numpy as np
import pandas as pd

import utils  # streamlit runs from root directory, so we can import utils directly

st.set_page_config(
    page_title="Minimum Multiplier", 
    page_icon=":times:",
    layout="wide",
)

def generate_plots(minimum_m_sweep_exchangerate_df, minimum_m_sweep_dealincome_df):
    st.markdown(
        """
        ## Minimum Quality Multiplier
        The following plot shows the minimum quality multiplier needed for the FIL+ strategy to be more 
        profitable than the CC strategy. This is simulated from a network perspective, rather than an individual SP perspective, and
        shown from two perspectives.

        In the first perspective, we compute the minimum multiplier needed as a function of the exchange rate, for user defined values of:
        - Deal Income
        - CC Sector Cost
        - Deal Sector Cost Multiplier

        In the second perspective, we compute the minimum multiplier needed as a function of Deal Income, for user defined values of:
        - Exchange Rate
        - CC Sector Cost
        - Deal Sector Cost Multiplier
        
        The slider bars on the left control these variables. Three lines are shown, for different cost scalings. Each line represents a scaling of the configured cost by the indicated value in the legend, in order to quickly
        see how changes in cost affect the required minimum multiplier.
"""
    )

    c1 = alt.Chart(minimum_m_sweep_exchangerate_df, title='Perspective 1 - Minimum Quality Multiplier').mark_line().encode(
        x=alt.X('exchange_rate:Q', title='Exchange Rate [$/FIL]'),
        y=alt.Y('minimum_m:Q', title='Multiplier'),
        color=alt.Color(
            'cost_scaling_str:N', 
            scale=alt.Scale(scheme='tableau20'),
            legend=alt.Legend(title='Cost Scaling')
        ),
    )
    st.altair_chart(c1, use_container_width=True)

    c2 = alt.Chart(minimum_m_sweep_dealincome_df, title='Perspective 2 - Minimum Quality Multiplier').mark_line().encode(
        x=alt.X('deal_income:Q', title='Deal Income [$/TiB/Yr]'),
        y=alt.Y('minimum_m:Q', title='Multiplier'),
        color=alt.Color(
            'cost_scaling_str:N', 
            scale=alt.Scale(scheme='tableau20'),
            legend=alt.Legend(title='Cost Scaling')
        ),
    )
    st.altair_chart(c2, use_container_width=True)

def compute_minimum_multiplier(scenario2erpt=None):
    exchange_rate_cfg =  st.session_state['mm_filprice_slider']
    onboarding_scenario = st.session_state['mm_onboarding_scenario'].lower()
    deal_income = st.session_state['mm_deal_income']
    cc_cost = st.session_state['mm_cc_cost']
    deal_cost_multiplier = st.session_state['mm_deal_cost_multiplier']

    erpt = scenario2erpt[onboarding_scenario]

    # sweep across a) exchange rate
    exchange_rate_vec = np.linspace(3, 50.0, 100)
    
    # perspective 1
    minimum_m_results = []
    for cost_scaling in [1.0, 0.8, 1.2]:
        for exchange_rate in exchange_rate_vec:
            cc_cost_full = cost_scaling*cc_cost
            cc_profit = erpt*exchange_rate - cc_cost_full
            # deal_profit = deal_income + m*erpt*exchange_rate - deal_cost_multiplier*cc_cost_full
            # deal_income + m*erpt*exchange_rate - deal_cost_multiplier*cc_cost_full = erpt_exchange_rate-cc_cost_full
            # => m = (cc_profit - deal_income + deal_cost_multiplier*cc_cost_full)/(erpt*exchange_rate)
            minimum_m = (cc_profit - deal_income + deal_cost_multiplier*cc_cost_full)/(erpt*exchange_rate)

            minimum_m_results.append({
                'cost_scaling': cost_scaling,
                'cost_scaling_str': f'{cost_scaling}x',
                'exchange_rate': exchange_rate,
                'minimum_m': minimum_m
            })
    minimum_m_sweep_exchangerate_df = pd.DataFrame(minimum_m_results)

    # perspective 2
    minimum_m_results = []
    # sweep across b) deal income
    deal_income_vec = np.linspace(0.00, 50.0, 100)
    for cost_scaling in [1.0, 0.8, 1.2]:
        for deal_income in deal_income_vec:
            cc_cost_full = cost_scaling*cc_cost
            cc_profit = erpt*exchange_rate_cfg - cc_cost_full
            # deal_profit = deal_income + m*erpt*exchange_rate - deal_cost_multiplier*cc_cost_full
            # deal_income + m*erpt*exchange_rate - deal_cost_multiplier*cc_cost_full = erpt_exchange_rate-cc_cost_full
            # => m = (cc_profit - deal_income + deal_cost_multiplier*cc_cost_full)/(erpt*exchange_rate)
            minimum_m = (cc_profit - deal_income + deal_cost_multiplier*cc_cost_full)/(erpt*exchange_rate_cfg)

            minimum_m_results.append({
                'cost_scaling': cost_scaling,
                'cost_scaling_str': f'{cost_scaling}x',
                'deal_income': deal_income,
                'minimum_m': minimum_m
            })
    minimum_m_sweep_dealincome_df = pd.DataFrame(minimum_m_results)

    generate_plots(minimum_m_sweep_exchangerate_df, minimum_m_sweep_dealincome_df)
        

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
        min_value=3., max_value=50., value=4.0, step=.1, format='%0.02f', key="mm_filprice_slider",
        on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
    )
    st.selectbox(
        'Onboarding Scenario', ('Status-Quo', 'Pessimistic', 'Optimistic'), key="mm_onboarding_scenario",
        on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
    )
    st.slider(
        'Deal Income ($/TiB/Yr)', 
        min_value=0.0, max_value=100.0, value=16.0, step=1.0, format='%0.02f', key="mm_deal_income",
        on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
    )
    st.slider(
        'CC Sector Cost ($/TiB/Yr)', 
        min_value=0.0, max_value=100.0, value=30.0, step=1.0, format='%0.02f', key="mm_cc_cost",
        on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
    )
    st.slider(
        'Deal Sector Cost Multiplier (X)', 
        min_value=1.0, max_value=10.0, value=4.0, step=1.0, format='%0.02f', key="mm_deal_cost_multiplier",
        on_change=compute_minimum_multiplier, kwargs=kwargs, disabled=False, label_visibility="visible"
    )
    st.button("Compute!", on_click=compute_minimum_multiplier, kwargs=kwargs, key="forecast_button")