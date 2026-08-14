#!/bin/bash
cd /home/alex/diabetes-cribado-pythonanywhere
source ~/.virtualenvs/diabetes-venv/bin/activate
streamlit run app.py --server.port=8000 --server.address=0.0.0.0

