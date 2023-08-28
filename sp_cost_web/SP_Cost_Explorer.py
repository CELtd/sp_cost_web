import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.write("# 👋 Welcome to Filecoin SP Cost Explorer!")

st.sidebar.success("Select a Page above.")

st.markdown(
    """
    ### Introduction
This web app enables an interactive exploration of how various costs and revenues affect the final net income of Storage Providers (SPs) in the Filecoin network. 

We examine five different SP strategies for participating in the Filecoin network:

1. FIL+: This models a FIL+ SP. Additional costs associated with being a FIL+ miner include: a) storing additional copies, b) increased bandwidth cost, c) business development costs, d) increased pledge collateral costs, and
2. FIL+ Cheat: This models an SP paying minimal business costs to pass the necessary hurdles to receive FIL+. They do not incur BizDev costs, but also do not receive revenue from clients.
3. FIL+ Exploit: This models an SP actively exploiting FIL+ and not growing a storage business. They don't incur BizDev costs, do not store an extra copy, have reduced bandwidth capabilities, but also do not receive revenue from clients.
4. CC: This models a CC SP. CC SPs have reduced costs associated with bandwidth, BizDev, pledge collateral, and data preparation, but also receive less block rewards.
5. Regular Deal (RD): This models an SP that is growing a storage business *without* FIL+. Their BizDev costs can be lower, and their pledge is lower, but they also receive less block rewards.

Two interactive calculators are provided. Both have slider bars that allow you to explore how different variables, such as the token exchange rate, costs associated with business development, network power onboarding rates, and data preparation costs, affect each SP strategy's net income. 

The first calculator (**Cost Breakdown**) graphically breaks down the different costs associated with each SP strategy. It also shows a bar graph of the expected profit, rank ordered by most rational strategy to least rational strategy.

The second calculator (**Rational Strategy**) explores how an individual variable affects the profit of each SP strategy. In this app, the plot's title indicates the variable being explored. All other variables which go into the cost calculation are held constant and set by the slider bar values.

In both calculators, the expected income from block rewards is computed using [MechaFil](https://github.com/protocol/mechafil-jax), a digital twin of the Filecoin economy. 

### Cost Computation
In the charts for both calculators, `net_income = revenue - costs`. The following table outlines all of the revenue and cost sources.  While most cost sources are adjustable via slider bar widgets, some are fixed due to their negligible impact on the overall cost.

Note that all revenue and costs are in units of $/TiB/Yr. 

|Revenue ($/TiB/Yr) |Fixed Costs ($/TiB/Yr)| Adjustable Costs ($/TiB/Yr)
|--|--|--|
|Block Rewards  |Sealing ($1.30) | Power Cost
|Deal Revenue  | Gas Cost w/ PSD ($2.30) | Bandwidth Cost
| | Gas Cost w/out PSD ($0.10) | Staff Cost
| | | Pledge (% of Block Rewards)
| | | Data Prep
| | | FIL+ BizDev
| | | RD BizDev
| | | Storing Extra Copy

  

### How to use this app

**👈 Select an App from the sidebar** to get started

### Want to learn more?

- Check out [CryptoEconLab](https://cryptoeconlab.io)

- Engage with us on [X](https://x.com/cryptoeconlab)
"""
)