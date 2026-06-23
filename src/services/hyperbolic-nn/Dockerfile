ARG BUILD_MODE=dev
FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

WORKDIR /usr/src

COPY src/services/hyperbolic-nn/requirements.txt /tmp/requirements.txt
RUN python -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

COPY src/services/hyperbolic-nn/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/usr/src
ENV ROOT_PATH=/usr/src
ENV OUTPUT_DIR=/jobs
ENV MPLBACKEND=Agg

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
