import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.write("# Welcome to Filecoin SP Cost Explorer! 👋")

st.sidebar.success("Select a Page above.")

st.markdown(
    """
    TODO: Describe (make sure you mention that it uses mechafil to compute expected BR).
    **👈 Select a page from the sidebar** to get staretd
    
    ### Want to learn more?
    - Check out [CryptoEconLab](https://cryptoeconlab.io)
    - Engage with us on [X](https://x.com/cryptoeconlab)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
"""
)