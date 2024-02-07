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
    "Revenues": ["Deal Income ($16/TiB/Yr)", "Block Rewards ($8/TiB/Yr)", None, None, None, None, None, None],
    "Protocol Costs": ["Gas ($2/TiB/Yr)", "Sealing ($2/TiB/Yr)", None, None, None, None, None, None],
    "Operational Costs": ["Power ($8/TiB/Yr)", "Bandwidth ($6/TiB/Yr)", "Staff ($8/TiB/Yr)", "Data Prep ($1/TiB/Yr)", None, None, None, None],
    "Financing Costs": ["Pledge Rev Share ($4/TiB/Yr)", None, None, None, None, None, None, None],
    "Additional Costs": ["Biz Dev ($8/TiB/Yr)", "Extra Copies ($2/TiB/Yr)", "Extra BW ($2/TiB/Yr)", None, None, None, None, None]
})

# Display the editable dataframe
st.title("Filecoin Storage Provider Survey")

st.write("The following is an anonymous survey to help understand costs and revenues as a storage provider. " + \
         "Responses will be summarized to provide insights into Storage-Provider microeconomics. " + \
         "Please provide your best estimate with as much detail as possible.  " + \
         "\n\n All fields are optional. Thank you for your time! " + \
         "\n\n Click submit on the 👈 to send your response.  \n\n")

# color this header blue
st.markdown("<h4 style='color: blue;'>Ledger</h4>", unsafe_allow_html=True)
st.markdown("""
This section requests information regarding costs and revenues. In the table below, each entry represents
a particular cost or revenue source for the category specified by the column. Default entries are shown below in units of USD/TiB/Yr, and can be edited by double clicking
on the cell. \n\n Feel free to use units that are easiest for you. Please aim to provide a description for each entry,
similar to the default entries. For example: Rather than entering a value of \$16/TiB/Yr, it is preferable to enter 
"Deal Income (\$16/TiB/Yr)", and so on. The default text descriptions and categories are provided as a guide.
\n\n
If you decide to provide aggregated costs and revenues, please indicate so with a description such as: Total($24/TiB/Yr). \n\n
 \n\n
""")
edited_df = st.data_editor(l3_df, use_container_width=True)

st.markdown("<h4 style='color: blue;'>Basic Information</h4>", unsafe_allow_html=True)
location = st.text_input("Geographic Location", placeholder="Quito, Ecuador", key="location")
rbp_size = st.text_input("RBP (PiB)", placeholder="10", key="rbp")
qap_size = st.text_input("QAP (PiB)", placeholder="50", key="qap")
feedback = st.text_area("Feedback", placeholder="Please provide any feedback or additional information here.", key="feedback")

def get_default(str_in, default="Unknown"):
    if str_in is None or str_in == "":
        return default
    return str_in

def submit_fn():
    ts = pd.Timestamp.now().isoformat()
    data_dict = {
        "location": get_default(location, "Unknown"),
        "rbp_size": get_default(rbp_size, "-1"),
        "qap_size": get_default(qap_size, "-1"),
        "feedback": get_default(feedback, "No feedback provided"),
        "ledger": edited_df.to_json(),
        "submit_time": ts
    }
    with tempfile.TemporaryDirectory() as tmpdirname:
        fp = f"{tmpdirname}/survey.json"
        with open(fp, 'w') as f:
            json.dump(data_dict, f)
        slack("#sp_survey", "New survey submission from @ Time=%s!" % (ts,), files=[fp])
    st.toast("Thank you for your submission!", icon="👏")

with st.sidebar:
    st.button("Submit!", on_click=submit_fn)