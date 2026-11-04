# Sandbox image for running the SenseNova code-video skill under Qwen Code with no outside access beyond the model endpoint.
# Contents: Node 24, Python with the skill's audio-free packages, ffmpeg (libx264, aac), Playwright with Chromium, @napi-rs/canvas, Qwen Code.
# edge-tts and faster-whisper are left out on purpose: edge-tts sends text to an outside speech service and faster-whisper downloads models.
FROM node:24-bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ffmpeg libsndfile1 fonts-dejavu-core fonts-liberation2 \
      ca-certificates git curl jq procps less \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install numpy scipy soundfile librosa
ENV PATH=/opt/venv/bin:$PATH
RUN mkdir -p /opt/pw && cd /opt/pw && npm init -y >/dev/null \
    && npm install playwright @napi-rs/canvas \
    && npx playwright install --with-deps chromium \
    && chmod -R a+rX /ms-playwright /opt/pw && rm -rf /var/lib/apt/lists/*
ARG QWEN_CODE_VERSION=0.26.0
RUN npm install -g @qwen-code/qwen-code@${QWEN_CODE_VERSION} && npm cache clean --force
RUN { echo "node $(node --version)"; echo "python $(python3 --version)"; ffmpeg -version | head -1; \
      echo "playwright $(node -p "require('/opt/pw/node_modules/playwright/package.json').version")"; \
      echo "canvas $(node -p "require('/opt/pw/node_modules/@napi-rs/canvas/package.json').version")"; \
      echo "qwen $(qwen --version)"; pip list 2>/dev/null | grep -i -E "^(numpy|scipy|soundfile|librosa) "; } > /opt/versions.txt
WORKDIR /work
