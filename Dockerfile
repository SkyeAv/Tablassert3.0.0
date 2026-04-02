FROM python:3.14-slim

RUN pip install --no-cache-dir "tablassert[full]"

ENTRYPOINT ["tablassert"]
CMD ["--help"]
