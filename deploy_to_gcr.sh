#!/bin/bash

gcloud builds submit --tag gcr.io/cel-streamlit/sp-cost-web

gcloud run deploy sp-cost-web \
  --image gcr.io/cel-streamlit/sp-cost-web \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances=0 \
  --port=8501
