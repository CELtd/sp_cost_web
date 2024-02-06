#!/bin/bash

source activate cel
streamlit run sp_cost_web/SP_Cost_Explorer.py --server.runOnSave True --server.allowRunOnSave True --server.headless True