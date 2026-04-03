FROM python:3.14-slim

RUN pip install --no-cache-dir "tablassert"

VOLUME ["/data", "/datassert"]

ENTRYPOINT ["tablassert"]
CMD ["--help"]
