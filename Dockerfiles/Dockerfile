FROM tensorflow/tensorflow:2.19.0-gpu-jupyter
# Enrique: FROM tensorflow/tensorflow:2.19.0-jupyter

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && pip install tensorflow[and-cuda]==2.19.0
# Enrique : RUN pip install --upgrade pip && \ pip install tensorflow==2.16.1

RUN pip install --no-cache-dir -r requirements.txt
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh
# Enrique: RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]