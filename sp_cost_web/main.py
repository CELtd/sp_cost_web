#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from datetime import date, timedelta

import time

import numpy as np
import pandas as pd
import jax.numpy as jnp

import streamlit as st
import streamlit.components.v1 as components
import st_debug as d
import altair as alt

import mechafil_jax.data as data
import mechafil_jax.sim as sim
import mechafil_jax.constants as C
import mechafil_jax.minting as minting
import mechafil_jax.date_utils as du

import scenario_generator.utils as u

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# local_css("debug.css")

@st.cache_data
def get_offline_data(start_date, current_date, end_date):
    PUBLIC_AUTH_TOKEN='Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ'
    offline_data = data.get_simulation_data(PUBLIC_AUTH_TOKEN, start_date, current_date, end_date)

    _, hist_rbp = u.get_historical_daily_onboarded_power(current_date-timedelta(days=180), current_date)
    _, hist_rr = u.get_historical_renewal_rate(current_date-timedelta(days=180), current_date)
    _, hist_fpr = u.get_historical_filplus_rate(current_date-timedelta(days=180), current_date)

    smoothed_last_historical_rbp = float(np.median(hist_rbp[-30:]))
    smoothed_last_historical_rr = float(np.median(hist_rr[-30:]))
    smoothed_last_historical_fpr = float(np.median(hist_fpr[-30:]))

    # run mechafil and compute expected block rewards. we only need to do this once
    scenarios = ['pessimistic', 'status-quo', 'optimistic']
    scenario_scalers = [0.5, 1.0, 1.5]

    forecast_length = (end_date-start_date).days
    sector_duration = 365
    lock_target = 0.3

    scenario2erpt = {}
    for ii, scenario_scaler in enumerate(scenario_scalers):    
        scenario = scenarios[ii]
        
        rbp = jnp.ones(forecast_length) * smoothed_last_historical_rbp * scenario_scaler
        rr = jnp.ones(forecast_length) * smoothed_last_historical_rr * scenario_scaler
        fpr = jnp.ones(forecast_length) * smoothed_last_historical_fpr
        
        simulation_results = sim.run_sim(
            rbp,
            rr,
            fpr,
            lock_target,
            start_date,
            current_date,
            forecast_length,
            sector_duration,
            offline_data
        )
        # scenario2results[scenario] = simulation_results
        expected_rewards_per_sector_today = float(simulation_results['1y_return_per_sector'][0])
    
        # extract the block-rewards per tib for each scenario
        sectors_per_tib = (1024**4) / C.SECTOR_SIZE
        brpt = expected_rewards_per_sector_today * sectors_per_tib
        scenario2erpt[scenario] = brpt
    
    return scenario2erpt

def compute_costs(scenario2erpt=None):
    filp_multiplier = 10

    onboarding_scenario = st.session_state['onboarding_scenario'].lower()
    erpt = scenario2erpt[onboarding_scenario]
    
    exchange_rate =  st.session_state['filprice_slider']
    borrowing_cost_pct = st.session_state['borrow_cost_pct'] / 100.0
    bd_cost_tib_per_yr = st.session_state['bizdev_cost']
    deal_income_tib_per_yr = st.session_state['deal_income']
    data_prep_cost_tib_per_yr = st.session_state['data_prep_cost']
    penalty_tib_per_yr = st.session_state['cheating_penalty']
    
    # Definitions (we can make these configurable later, potentially)
    sealing_costs_tib_per_yr = 1.3

    gas_cost_tib_per_yr = (2250.+108.)/1024.
    gas_cost_without_psd_tib_per_yr = 108./1024.
    power_cost_tib_per_yr = 6000/1024.0
    bandwidth_10gbps_tib_per_yr = 6600/1024.0
    bandwidth_1gbps_tib_per_yr = 660/1024.0
    staff_cost_tib_per_yr = 9830.0/1024.0  # $10k/yr/TiB

    # create a dataframe for each of the miner profiles
    filp_miner = {
        'SP Type': 'FIL+',
        'block_rewards': erpt*exchange_rate*filp_multiplier,
        'deal_income': deal_income_tib_per_yr,
        'pledge_cost': erpt*exchange_rate*filp_multiplier*borrowing_cost_pct,
        'gas_cost': gas_cost_tib_per_yr,
        'power_cost': power_cost_tib_per_yr,
        'bandwidth_cost': bandwidth_10gbps_tib_per_yr,
        'staff_cost': staff_cost_tib_per_yr,
        'sealing_cost': sealing_costs_tib_per_yr,
        'data_prep_cost': data_prep_cost_tib_per_yr,
        'bd_cost': bd_cost_tib_per_yr,
        'extra_copy_cost': (staff_cost_tib_per_yr+bd_cost_tib_per_yr+bandwidth_10gbps_tib_per_yr)*0.9,
        'cheating_cost': 0
    }
    rd_miner = {
        'SP Type': 'Regular Deal',
        'block_rewards': erpt*exchange_rate,
        'deal_income': deal_income_tib_per_yr,
        'pledge_cost': erpt*exchange_rate*borrowing_cost_pct,
        'gas_cost': gas_cost_tib_per_yr,
        'power_cost': power_cost_tib_per_yr,
        'bandwidth_cost': bandwidth_10gbps_tib_per_yr,
        'staff_cost': staff_cost_tib_per_yr,
        'sealing_cost': sealing_costs_tib_per_yr,
        'data_prep_cost': data_prep_cost_tib_per_yr,
        'bd_cost': bd_cost_tib_per_yr/10.0,
        'extra_copy_cost': (staff_cost_tib_per_yr+bd_cost_tib_per_yr/10.0+bandwidth_10gbps_tib_per_yr)*0.9,
        'cheating_cost': 0
    }
    filp_cheat_miner = {
        'SP Type':'FIL+ Cheat',
        'block_rewards': erpt*exchange_rate*filp_multiplier,
        'deal_income': 0,
        'pledge_cost': erpt*exchange_rate*filp_multiplier*borrowing_cost_pct,
        'gas_cost': gas_cost_tib_per_yr,
        'power_cost': power_cost_tib_per_yr,
        'bandwidth_cost': bandwidth_1gbps_tib_per_yr,
        'staff_cost': staff_cost_tib_per_yr,
        'sealing_cost': sealing_costs_tib_per_yr,
        'data_prep_cost': 1,
        'bd_cost': 0,
        'extra_copy_cost': 0,
        'cheating_cost': penalty_tib_per_yr
    }
    cc_miner = {
        'SP Type':'CC',
        'block_rewards': erpt*exchange_rate,
        'deal_income': 0,
        'pledge_cost': erpt*exchange_rate*borrowing_cost_pct,
        'gas_cost': gas_cost_without_psd_tib_per_yr,
        'power_cost': power_cost_tib_per_yr,
        'bandwidth_cost': bandwidth_1gbps_tib_per_yr,
        'staff_cost': staff_cost_tib_per_yr,
        'sealing_cost': sealing_costs_tib_per_yr,
        'data_prep_cost': 0,
        'bd_cost': 0,
        'extra_copy_cost': 0,
        'cheating_cost': 0
    }
    aws = {
        'SP Type':'AWS',
        'block_rewards': 0,
        'deal_income': 6.6,
        'pledge_cost': 0,
        'gas_cost': 0,
        'power_cost': 0,
        'bandwidth_cost': 0,
        'staff_cost': 0,
        'sealing_cost': 0,
        'data_prep_cost': 0,
        'bd_cost': 0,
        'extra_copy_cost': 0,
        'cheating_cost': 0
    }
    df = pd.DataFrame([filp_miner, rd_miner, filp_cheat_miner, cc_miner, aws])
    # add final accounting to the DF
    df['revenue'] = df['block_rewards'] + df['deal_income']
    df['cost'] = df['pledge_cost'] + df['gas_cost'] + df['power_cost'] + df['bandwidth_cost'] + df['staff_cost'] + df['sealing_cost'] + df['data_prep_cost'] + df['bd_cost'] + df['extra_copy_cost'] + df['cheating_cost']
    df['cost'] = df['cost']*-1
    df['profit'] = df['revenue'] + df['cost']

    # st.dataframe(df.T)
    plot_costs(df)

def plot_costs(df):
    # sort by profit
    # df_sorted = df.sort_values(by=['profit'], ascending=False).reset_index(drop=True)
    # df_sorted['revenue'] = df_sorted['revenue'].astype(float)
    # df_sorted['cost'] = df_sorted['cost'].astype(float)
    # df_sorted['profit'] = df_sorted['profit'].astype(float)

    # st.bar_chart(data=df_sorted, x='SP Type', y=['revenue', 'cost'])
    acounting_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('SP Type', sort='-y'),
        y=alt.Y('profit'),
        color=alt.Color('SP Type', scale=alt.Scale(scheme='tableau20'))
        # color=alt.condition(
        #     alt.datum.profit > 0,
        #     alt.value("green"),  # The positive color
        #     alt.value("red")  # The negative color
        # )
    )
    st.altair_chart(acounting_chart, use_container_width=True)
    # st.dataframe(df.T)
    
    dft = df.T
    st.table(dft)
    # Custom formatting
    st.table(dft.style.format({"{:.2f}"}))
    
def main():
    st.set_page_config(
        page_title="Cost Explorer",
        page_icon="🚀",  # TODO: can update this to the FIL logo
        layout="wide",
    )
    
    current_date = date.today() - timedelta(days=3)
    mo_start = min(current_date.month - 1 % 12, 1)
    start_date = date(current_date.year, mo_start, 1)
    forecast_length_days=365*3
    end_date = current_date + timedelta(days=forecast_length_days)
    scenario2erpt = get_offline_data(start_date, current_date, end_date)

    with st.sidebar:
        st.title('SP Cost Scenario Explorer')

        compute_costs_kwargs = {
            'scenario2erpt':scenario2erpt
        }
        st.slider(
            "FIL Exchange Rate ($/FIL)", 
            min_value=3., max_value=50., value=4.0, step=.1, format='%0.02f', key="filprice_slider",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Borrowing Costs (Pct. of Pledge)', 
            min_value=0.0, max_value=100.0, value=50.0, step=1.00, format='%0.02f', key="borrow_cost_pct",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Biz Dev Cost (TiB/Yr)', 
            min_value=5.0, max_value=50.0, value=34.0, step=1.0, format='%0.02f', key="bizdev_cost",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Deal Income ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=16.0, step=1.0, format='%0.02f', key="deal_income",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Data Prep Cost ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=1.0, step=1.0, format='%0.02f', key="data_prep_cost",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.slider(
            'Cheating Penalty ($/TiB/Yr)', 
            min_value=0.0, max_value=50.0, value=0.0, step=1.0, format='%0.02f', key="cheating_penalty",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
        st.selectbox(
            'Onboarding Scenario', ('Pessimistic', 'Status-Quo', 'Optimistic'), key="onboarding_scenario",
            on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
        )
    
    if "debug_string" in st.session_state:
        st.markdown(
            f'<div class="debug">{ st.session_state["debug_string"]}</div>',
            unsafe_allow_html=True,
        )
    components.html(
        d.js_code(),
        height=0,
        width=0,
    )

if __name__ == '__main__':
    main()