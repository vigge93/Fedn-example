FROM ghcr.io/scaleoutsystems/fedn/fedn:latest

WORKDIR /app

# Install dependencies
COPY requirements.txt ./

RUN /venv/bin/pip uninstall fedn -y
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt
RUN /venv/bin/pip install fedn

WORKDIR /app

ENTRYPOINT ["/venv/bin/fedn"]
