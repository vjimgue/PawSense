#!/bin/bash
# Start Jupyter Notebook in the background
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root &

# Start Streamlit in the foreground
# (The container stays alive as long as this process runs)
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0