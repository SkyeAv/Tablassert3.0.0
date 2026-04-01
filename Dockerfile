FROM python:3.14-slim

RUN pip install --no-cache-dir tablassert

ENTRYPOINT ["tablassert"]
CMD ["--help"]
