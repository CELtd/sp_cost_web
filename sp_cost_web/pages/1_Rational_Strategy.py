import streamlit as st

st.set_page_config(page_title="Rational Strategy", page_icon=":brain:")

# with st.sidebar:
#     st.slider(
#         "FIL Exchange Rate ($/FIL)", 
#         min_value=3., max_value=50., value=4.0, step=.1, format='%0.02f', key="filprice_slider",
#         on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
#     )
#     st.selectbox(
#         'Onboarding Scenario', ('Status-Quo', 'Pessimistic', 'Optimistic'), key="onboarding_scenario",
#         on_change=compute_costs, kwargs=compute_costs_kwargs, disabled=False, label_visibility="visible"
#     )      