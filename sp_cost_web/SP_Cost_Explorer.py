import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.markdown("[![CryptoEconLab](./app/static/cover.png)](https://cryptoeconlab.io)")

st.sidebar.success("Select a Page above.")

st.markdown(
    """
    ### Filecoin SP Cost Explorer
This web app enables an interactive exploration of how various costs and revenues affect the final net income of Storage Providers (SPs) in the Filecoin network. 

We examine six different SP strategies for participating in the Filecoin network:

1. FIL+: This models a FIL+ SP. Additional costs associated with being a FIL+ miner include: a) storing additional copies, b) increased bandwidth cost, c) business development costs, d) increased pledge collateral costs, and
2. Regular Deal (RD): This models an SP that is growing a storage business *without* FIL+. Their BizDev and bandwidth costs can be lower, and their pledge is lower, but they also receive less block rewards.
3. CC: This models a CC SP. CC SPs have reduced costs associated with bandwidth (10% of FIL+), BizDev, pledge collateral, and data preparation, but also receive less block rewards.
4. V1-ExploitFIL+: This models an SP actively exploiting FIL+ and not growing a storage business. They don't incur BizDev costs, do not store an extra copy, have reduced bandwidth costs(10% of FIL+) and are not subject to any penalties for exploiting FIL+. They also do not receive revenue from clients.
4. V2-ExploitFIL+: The primary difference between `V1` and `V2` is that in `V2`, the SP pays for storing an extra copy and has enough bandwidth to pass any retrieval tests that may be conducted.
4. V3-ExploitFIL+: The primary difference between `V2` and `V3` is that in `V3`, the SP is subject to slashing costs if they fail to pass integrity tests for FIL+.

Three interactive calculators are provided. All have slider bars that allow you to explore how different variables, such as the token exchange rate, costs associated with business development, network power onboarding rates, and data preparation costs, affect each SP strategy's net income. 

The first calculator (**Cost Breakdown**) graphically breaks down the different costs associated with each SP strategy. It also shows a bar graph of the expected profit, rank ordered by most rational strategy to least rational strategy.

The second calculator (**Rational Strategy**) explores how an individual variable affects the profit of each SP strategy. In this app, the plot's title indicates the variable being explored. All other variables which go into the cost calculation are held constant and set by the slider bar values.

The third calculator (**Minimum Multiplier**) explores the necessary minimum QA multiplier needed for the FIL+ strategy to be more profitable than the CC strategy. This is simulated at a network level, so the variables to control costs are aggregated into a single cost for CC sectors, and a multiplier to scale from the CC sector cost to the Deal sector cost.

In all calculators, the expected income from block rewards (in units of FIL) is computed using [MechaFil](https://github.com/protocol/mechafil-jax), a digital twin of the Filecoin economy. The exchange rate slider then converts FIL to USD.

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

- Read more of our research on [Medium](https://medium.com/cryptoeconlab) and [HackMD](https://hackmd.io/@cryptoecon/almanac/)

### Disclaimer
CryptoEconLab designed this application for informational purposes only. CryptoEconLab does not provide legal, tax, financial or investment advice. No party should act in reliance upon, or with the expectation of, any such advice.
"""
)