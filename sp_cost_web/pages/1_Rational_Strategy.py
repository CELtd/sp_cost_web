import streamlit as st
import altair as alt

from datetime import date, timedelta
import numpy as np
import pandas as pd

import utils  # streamlit runs from root directory, so we can import utils directly

st.set_page_config(page_title="Rational Strategy", page_icon=":brain:")

def generate_plots(borrowing_cost_df):
    col1, col2 = st.columns(2)

    with col1:
        borrowing_cost_chart = alt.Chart(borrowing_cost_df, title="Borrowing Cost").mark_rect().encode(
            alt.X("borrowing_cost_pct:N").title("Borrowing Cost Pct").axis(labelAngle=0),  # why does format=%0.02f in axis(...) not work?
            alt.Y("SP Type:O").title("Strategy"),
            alt.Color("rank:N").title(None),
            tooltip=[
                alt.Tooltip('SP Type', title='Strategy'),
                alt.Tooltip('rank', title='Rank')
            ]
        )
        st.altair_chart(borrowing_cost_chart, use_container_width=True)

def generate_rankings(scenario2erpt=None):
    # get fixed costs
    filp_multiplier = st.session_state['rs_filp_multiplier']
    rd_multiplier = st.session_state['rs_rd_multiplier']
    cc_multiplier = st.session_state['rs_cc_multiplier']

    onboarding_scenario = st.session_state['rs_onboarding_scenario'].lower()
    
    exchange_rate =  st.session_state['rs_filprice_slider']
    borrowing_cost_pct = st.session_state['rs_borrow_cost_pct'] / 100.0
    bd_cost_tib_per_yr = st.session_state['rs_bizdev_cost']
    deal_income_tib_per_yr = st.session_state['rs_deal_income']
    data_prep_cost_tib_per_yr = st.session_state['rs_data_prep_cost']
    penalty_tib_per_yr = st.session_state['rs_cheating_penalty']

    sp_profile_to_integer = {
        'FIL+ Exploit': 0,
        'FIL+': 1,
        'FIL+ Cheat': 2,
        'CC': 3,
        'Regular Deal': 4,
    }

    # sweep borrowing_cost, fix other costs
    borrowing_cost_vec = np.linspace(0,100,25)
    borrowing_cost_plot_vec = {}
    for borrowing_cost_sweep_pct in borrowing_cost_vec:
        borrowing_cost_sweep_frac = borrowing_cost_sweep_pct/100.0
        df = utils.compute_costs(scenario2erpt=scenario2erpt,
                                filp_multiplier=filp_multiplier, rd_multiplier=rd_multiplier, cc_multiplier=cc_multiplier,
                                onboarding_scenario=onboarding_scenario,
                                exchange_rate=exchange_rate, borrowing_cost_pct=borrowing_cost_sweep_frac,
                                bd_cost_tib_per_yr=bd_cost_tib_per_yr, deal_income_tib_per_yr=deal_income_tib_per_yr,
                                data_prep_cost_tib_per_yr=data_prep_cost_tib_per_yr, penalty_tib_per_yr=penalty_tib_per_yr)
        df.sort_values(by='profit', ascending=False, inplace=True)
        df['rank'] = df['SP Type'].apply(lambda x: sp_profile_to_integer[x])
        df['borrowing_cost_pct'] = borrowing_cost_sweep_pct
        borrowing_cost_plot_vec.append(df[['SP Type', 'rank', 'borrowing_cost_pct', 'profit']])
    borrowing_cost_plot_df = pd.concat(borrowing_cost_plot_vec)

    # sweep deal_income

    # sweep data_prep_cost

    # sweep bizdev_cost

    # plot
    generate_plots(borrowing_cost_plot_df)

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
    with st.expander("Cost Settings", expanded=False):
        st.slider(
            'Borrowing Costs (Pct. of Pledge)', 
            min_value=0.0, max_value=100.0, value=50.0, step=1.00, format='%0.02f', key="rs_borrow_cost_pct",
            on_change=generate_rankings, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Biz Dev Cost (TiB/Yr)', 
            min_value=5.0, max_value=50.0, value=8.0, step=1.0, format='%0.02f', key="rs_bizdev_cost",
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