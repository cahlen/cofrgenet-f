FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY . .

CMD ["bash"]
