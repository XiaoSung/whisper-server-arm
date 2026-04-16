# NGC 25.01 支援 GB10 (Grace Blackwell / DGX Spark)
FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    ffmpeg git cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir pybind11 && \
    git clone --recursive https://github.com/OpenNMT/CTranslate2.git /tmp/ct2 && \
    mkdir /tmp/ct2/build && \
    cd /tmp/ct2/build && \
    cmake .. \
        -DWITH_CUDA=ON \
        -DCUDA_DYNAMIC_LOADING=ON \
        -DOPENMP_RUNTIME=COMP \
        -DWITH_MKL=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd /tmp/ct2/python && \
    pip3 install --no-cache-dir . && \
    rm -rf /tmp/ct2

RUN pip3 install --no-cache-dir \
    "numpy<2" \
    fastapi \
    uvicorn \
    aiofiles \
    python-multipart \
    httpx \
    huggingface_hub \
    tokenizers \
    onnxruntime \
    av

RUN pip3 install --no-cache-dir --no-deps faster-whisper

COPY app/ ./

RUN mkdir -p /app/uploads /app/outputs && chmod 777 /app/uploads /app/outputs

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
