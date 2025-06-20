FROM python:3.13
WORKDIR /user/local/tablassert
COPY . .
RUN pip install poetry
RUN poetry install
RUN poetry run tui