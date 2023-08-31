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

def generate_plots(minimum_m_df):
    st.markdown(
        """
        ## Minimum Quality Multiplier
        The following plot shows the minimum quality multiplier needed for the FIL+ strategy to be more 
        profitable than the CC strategy. This is simulated from a network perspective, rather than an individual SP perspective.
        
        The slider bars on the left control the cost of the CC sector, and the multiplier to scale from the CC sector cost to the Deal sector cost.
        Additional variables that can be controlled include the exchange rate, and the expected income from deals per TiB.

        Three lines are shown, for different cost scalings. Each line represents a scaling of the configured cost by the indicated value in the legend, in order to quickly
        see how changes in cost affect the required minimum multiplier.
"""
    )

    c = alt.Chart(minimum_m_df, title='Minimum Quality Multiplier').mark_line().encode(
        x=alt.X('exchange_rate:Q', title='Exchange Rate [$/FIL]'),
        y=alt.Y('minimum_m:Q', title='Multiplier'),
        color=alt.Color(
            'cost_scaling_str:N', 
            scale=alt.Scale(scheme='tableau20'),
            legend=alt.Legend(title='Cost Scaling')
        ),
    )
    st.altair_chart(c, use_container_width=True)

def compute_minimum_multiplier(scenario2erpt=None):
    onboarding_scenario = st.session_state['mm_onboarding_scenario'].lower()
    deal_income = st.session_state['mm_deal_income']
    cc_cost = st.session_state['mm_cc_cost']
    deal_cost_multiplier = st.session_state['mm_deal_cost_multiplier']

    erpt = scenario2erpt[onboarding_scenario]

    # sweep across a) exchange rate
    exchange_rate_vec = np.linspace(3, 50.0, 100)
    
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