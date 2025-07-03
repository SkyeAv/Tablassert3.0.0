FROM python:3.13

WORKDIR /usr/local/tablassert

EXPOSE 8080

COPY pyproject.toml poetry.lock* /usr/local/tablassert/
COPY src /usr/local/tablassert/src

ENV POETRY_VERSION=2.1.3

RUN pip install "poetry==$POETRY_VERSION"
RUN poetry install

ENTRYPOINT ["poetry", "run", "cli"]