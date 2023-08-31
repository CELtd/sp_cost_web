import streamlit as st
import altair as alt

from collections import OrderedDict, defaultdict

from datetime import date, timedelta
import numpy as np
import pandas as pd

from numpyro import distributions as dist
from numpyro import sample
from jax import random
import matplotlib.pyplot as plt
import pandas as pd

import random as pyrandom

import utils  # streamlit runs from root directory, so we can import utils directly

def plot_rankings(strategy2ranking):
    x = pd.DataFrame(strategy2ranking).T[[1,2,3,4,5,6]].fillna(1)
    x['SP Type'] = x.index
    x.index = np.arange(len(x))
    z = pd.melt(x, id_vars=['SP Type'])
    ch = alt.Chart(z, width=alt.Step(20), title="Monte-Carlo Ranking").mark_bar().encode(
        x=alt.X('variable:N', title='Rank', axis=alt.Axis(labelAngle=0),),
        y=alt.Y('value:Q', title='Counts'),
        xOffset="SP Type:N",
        color=alt.Color("SP Type:N", scale=alt.Scale(scheme='tableau20'))
    ).configure_axis(
        labelAngle=0,
        labelFontSize=20,
        titleFontSize=20
    )
    st.altair_chart(ch, use_container_width=True)

def run_mc_sim(scenario2erpt=None):
    n_samples = 1000  # TODO: revisit

    exchange_rate = st.session_state['mc_filprice_slider']
    onboarding_scenario = st.session_state['mc_onboarding_scenario'].lower()
    
    filp_multiplier = st.session_state['mc_filp_multiplier']
    rd_multiplier = st.session_state['mc_rd_multiplier']
    cc_multiplier = st.session_state['mc_cc_multiplier']

    client_fees_lambda = 1.0/st.session_state['mc_deal_income']
    staff_fees_alpha = st.session_state['mc_staff']
    data_prep_alpha = st.session_state['mc_data_prep']
    bizdev_lambda = 1.0/st.session_state['mc_bizdev']
    extra_copy_alpha = st.session_state['mc_extracopy']
    bandwidth_alpha = st.session_state['mc_bw']
    power_alpha = st.session_state['mc_power']
    gamma_alpha = st.session_state['gamma_alpha']

    seed = pyrandom.randint(0, 2**24)

    client_fees = sample('x', dist.Exponential(client_fees_lambda).expand([n_samples]), rng_key=random.PRNGKey(seed))
    staff = sample('x', dist.Gamma(staff_fees_alpha,gamma_alpha).expand([n_samples]), rng_key=random.PRNGKey(seed+1))
    data_prep = sample('x', dist.Gamma(data_prep_alpha,gamma_alpha).expand([n_samples]), rng_key=random.PRNGKey(seed+2))
    bd = sample('x', dist.Exponential(bizdev_lambda).expand([n_samples]), rng_key=random.PRNGKey(seed+3))
    extra_copy = sample('x', dist.Gamma(extra_copy_alpha,gamma_alpha).expand([n_samples]), rng_key=random.PRNGKey(seed+4))
    bandwidth = sample('x', dist.Gamma(bandwidth_alpha,gamma_alpha).expand([n_samples]), rng_key=random.PRNGKey(seed+5))
    power_and_colo = sample('x', dist.Gamma(power_alpha,gamma_alpha).expand([n_samples]), rng_key=random.PRNGKey(seed+6))
    
    filp_profile = pd.DataFrame({
        'client_fees': client_fees,
        'staff': staff,
        'data_prep': data_prep,
        'bd': bd,
        'extra_copy': extra_copy,
        'bandwidth': bandwidth,
        'power_and_colo': power_and_colo   
    })

    borrowing_cost = st.session_state['mc_borrow_cost_pct']/100.0

    strategy2ranking = {}
    for _, row in filp_profile.iterrows():
        cost_df = utils.compute_costs(
            scenario2erpt=scenario2erpt, 
            filp_multiplier=filp_multiplier, rd_multiplier=rd_multiplier, cc_multiplier=cc_multiplier,
            onboarding_scenario=onboarding_scenario,
            exchange_rate=exchange_rate, borrowing_cost_pct=borrowing_cost,
            filp_bd_cost_tib_per_yr=row['bd'], rd_bd_cost_tib_per_yr=row['bd']*0.5,
            deal_income_tib_per_yr=row['client_fees'],
            data_prep_cost_tib_per_yr=row['data_prep'], penalty_tib_per_yr=0.0,
            power_cost_tib_per_yr=row['power_and_colo'], 
            bandwidth_10gbps_tib_per_yr=row['bandwidth'], 
            staff_cost_tib_per_yr=row['staff']
        )
        # rank by profit
        cost_df['rank'] = cost_df['profit'].rank(ascending=False, method='first').astype(int)
        sp_types = cost_df['SP Type'].values
        ranks = cost_df['rank'].values
        for sp_type, rank in zip(sp_types, ranks):
            if sp_type not in strategy2ranking:
                strategy2ranking[sp_type] = defaultdict(int)
            strategy2ranking[sp_type][rank] += 1

    plot_rankings(strategy2ranking)
    

st.set_page_config(
    page_title="Monte Carlo Simulation", 
    #page_icon=":brain:",
    layout="wide",
)

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
        min_value=3., max_value=50., value=4.0, step=.1, format='%0.02f', key="mc_filprice_slider",
        on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
    )
    st.selectbox(
        'Onboarding Scenario', ('Status-Quo', 'Pessimistic', 'Optimistic'), key="mc_onboarding_scenario",
        on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
    )
    st.slider(
        'Borrowing Costs (Pct. of Pledge)', 
        min_value=0.0, max_value=100.0, value=50.0, step=1.00, format='%0.02f', key="mc_borrow_cost_pct",
        on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
    )

    with st.expander("Distribution Settings", expanded=False):
        st.slider(
            "Client Fees (Mean)", 
            min_value=1.0, max_value=40., value=16.0, step=.1, format='%0.02f', key="mc_deal_income",
            on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            "Staff Fees (alpha)", 
            min_value=1.0, max_value=50., value=16.0, step=.1, format='%0.02f', key="mc_staff",
            on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            "Data Prep (alpha)", 
            min_value=0.1, max_value=50., value=2.0, step=.1, format='%0.02f', key="mc_data_prep",
            on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            "Biz Dev (Mean)", 
            min_value=1.0, max_value=50.0, value=16.0, step=.1, format='%0.02f', key="mc_bizdev",
            on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            "Extra Copy (alpha)", 
            min_value=0.1, max_value=50., value=14.0, step=.1, format='%0.02f', key="mc_extracopy",
            on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            "Bandwidth (alpha)", 
            min_value=0.1, max_value=50., value=12.0, step=.1, format='%0.02f', key="mc_bw",
            on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            "Power+COLO (alpha)", 
            min_value=0.1, max_value=50., value=12.0, step=.1, format='%0.02f', key="mc_power",
            on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            "Gamma Rate [all] (Beta)", 
            min_value=0.1, max_value=10., value=2.0, step=.1, format='%0.02f', key="gamma_alpha",
            on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
    
    with st.expander("Multipliers", expanded=False):
        st.slider(
            'CC', min_value=1, max_value=20, value=1, step=1, key="mc_cc_multiplier",
            on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'RD', min_value=1, max_value=20, value=1, step=1, key="mc_rd_multiplier",
            on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'FIL+', min_value=1, max_value=20, value=10, step=1, key="mc_filp_multiplier",
            on_change=run_mc_sim, kwargs=kwargs, disabled=False, label_visibility="visible"
        )
    st.button("Compute!", on_click=run_mc_sim, kwargs=kwargs, key="forecast_button")