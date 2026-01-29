#!/bin/bash

PROJECT=cel-streamlit

gcloud builds submit --tag gcr.io/$PROJECT/sp-cost-web --project $PROJECT

gcloud run deploy sp-cost-web \
  --image gcr.io/$PROJECT/sp-cost-web \
  --project $PROJECT \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances=0 \
  --port=8501
