FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
WORKDIR /app
RUN pip install --no-cache-dir --break-system-packages streamlit
COPY . .
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true", "--server.fileWatcherType=poll"]