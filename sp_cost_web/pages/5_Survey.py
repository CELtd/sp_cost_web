import streamlit as st
import pandas as pd
import json

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import tempfile

def slack(channel, msg, files=None):
    oauth_token = st.secrets["slack_token"]
    client = WebClient(token=oauth_token)
    client.chat_postMessage(channel=channel, text=msg)

    if files is not None:
        for fp in files:
            try:
                response = client.files_upload(
                    channels=channel,
                    file=fp,
                )
                assert response["file"]  # the uploaded file
            except SlackApiError as e:
                # You will get a SlackApiError if "ok" is False
                assert e.response["ok"] is False
                assert e.response["error"]

# Load your dataframe (you can replace this with your own data)
l3_df = pd.DataFrame({
    "Revenues": ["Deal Income", "Block Rewards", None, None],
    "Protocol": ["Gas", "Sealing", None, None],
    "Operational": ["Power", "Bandwidth", "Staff", "Data Prep"],
    "Financing": [None]*4,
    "Additional Costs": ["Biz Dev", "Extra Copies", "Extra BW", None]
})

l2_df = pd.DataFrame({
    "Revenues": [None],
    "Protocol": [None],
    "Operational": [None],
    "Financing": [None],
    "Additional Costs": [None]
})

l1_df = pd.DataFrame({
    "Revenues": [None],
    "Costs": [None],
})

# Display the editable dataframe
st.title("Filecoin Storage Provider Survey")

st.write("The following is a survey to help us understand your costs and revenues as a storage provider. " + \
         "Please answer the following questions honestly and to the best of your knowledge. Although we prefer to have " + \
         "as detailed of information as possible, we understand that some questions may be difficult to answer. In those cases, " + \
         "please provide your best estimate with as much detail as you are able to.  \n\n All fields are optional. Thank you for your time!")

# color this header blue
st.markdown("<h4 style='color: blue;'>Ledger</h4>", unsafe_allow_html=True)
st.markdown("""
This section requests information regarding costs and revenues. We have three levels of granularity:
- **L3** is the most granular
- **L2** is mid-level
- **L1** is the least granular

Select the level that best suits your needs. \n\n Double click on the cells and enter a numeric value.  The units of each entry are $/TiB/Yr.
""")
option = st.radio("", ["L1", "L2", "L3"],  horizontal=True, index=2)
if option == "L1":
    edited_df = st.data_editor(l1_df, use_container_width=True)
elif option == "L2":
    edited_df = st.data_editor(l2_df, use_container_width=True)
else:
    edited_df = st.data_editor(l3_df, use_container_width=True)

st.markdown("<h4 style='color: blue;'>Basic Information</h4>", unsafe_allow_html=True)
sp_name = st.text_input("Storage Provider Name", placeholder="DSA", key="sp_name")
miner_id = st.text_input("Miner ID", placeholder="0x9999999", key="miner_id")
location = st.text_input("Geographic Location", placeholder="Quito, Ecuador", key="location")
rbp_size = st.text_input("RBP (PiB)", placeholder="10", key="rbp")
qap_size = st.text_input("QAP (PiB)", placeholder="50", key="qap")

st.markdown("<h4 style='color: blue;'>Storage Infrastructure</h4>", unsafe_allow_html=True)
storage_setup = st.text_input("Describe your storage setup", placeholder="hardware, data centers, etc.", key="storage_setup")
redundancy = st.text_input("How do you manage redundancy and durability?", placeholder="We are very redundant", key="redundancy")
challenges = st.text_input("What are some challenges faced in maintaining and scaling your storage infrastructure?", placeholder="We are very challenged", key="challenges")

st.markdown("<h4 style='color: blue;'>Deals and Contracts</h4>", unsafe_allow_html=True)
deal_setup = st.text_input("How do you find clients and negotiate storage deals?", placeholder="We know people", key="deal_setup")
deal_challenges = st.text_input("What are some challenges faced in finding and negotiating storage deals?", placeholder="We aren't challenged", key="deal_challenges")
deal_pain_points = st.text_input("What are some pain points in dealing with clients or other network participants?", placeholder="We have no pain", key="deal_pain_points")

st.markdown("<h4 style='color: blue;'>Network Utilization</h4>", unsafe_allow_html=True)
utilization = st.text_input("How often are your storage resources fully utilized?", placeholder="We are always full", key="utilization")
utilization_factors = st.text_input("What factors affect your storage utilization?", placeholder="We are affected by many factors", key="utilization_factors")
utilization_strategies = st.text_input("What strategies do you use to improve storage utilization?", placeholder="We use many strategies", key="utilization_strategies")

st.markdown("<h4 style='color: blue;'>Filecoin Token (FIL)</h4>", unsafe_allow_html=True)
volatility = st.text_input("How do you handle FIL volatility?", placeholder="We are very volatile", key="volatility")
br = st.text_input("Do you rely solely on block rewards, or do you explore other revenue streams?", placeholder="We rely on block rewards", key="br")

def get_default(str_in, default="Unknown"):
    if str_in is None or str_in == "":
        return default
    return str_in

def submit_fn():
    ts = pd.Timestamp.now().isoformat()
    data_dict = {
        "sp_name": get_default(sp_name, "Anonymous"),
        "miner_id": get_default(miner_id, "0x0000000"),
        "location": get_default(location, "Unknown"),
        "rbp_size": get_default(rbp_size, "-1"),
        "qap_size": get_default(qap_size, "-1"),
        "storage_setup": get_default(storage_setup, "Unknown"),
        "redundancy": get_default(redundancy, "Unknown"),
        "challenges": get_default(challenges, "Unknown"),
        "deal_setup": get_default(deal_setup, "Unknown"),
        "deal_challenges": get_default(deal_challenges, "Unknown"),
        "deal_pain_points": get_default(deal_pain_points, "Unknown"),
        "utilization": get_default(utilization, "Unknown"),
        "utilization_factors": get_default(utilization_factors, "Unknown"),
        "utilization_strategies": get_default(utilization_strategies, "Unknown"),
        "volatility": get_default(volatility, "Unknown"),
        "br": get_default(br, "Unknown"),
        "ledger_level": option,
        "ledger": edited_df.to_json(),
        "submit_time": ts
    }
    with tempfile.TemporaryDirectory() as tmpdirname:
        fp = f"{tmpdirname}/survey.json"
        with open(fp, 'w') as f:
            json.dump(data_dict, f)
        slack("#sp_survey", "New survey submission from SP:%s ID:%s @ Time=%s!" % (sp_name, miner_id, ts), files=[fp])
    st.toast("Thank you for your submission!", icon="👏")

with st.sidebar:
    st.button("Submit!", on_click=submit_fn)