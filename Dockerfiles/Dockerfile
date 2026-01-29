FROM tensorflow/tensorflow:2.19.0-gpu-jupyter
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install tensorflow[and-cuda]==2.19.0
RUN pip install --no-cache-dir -r requirements.txt
# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# Set the entrypoint
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]