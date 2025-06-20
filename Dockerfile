FROM python:3.13
WORKDIR /user/local/tablassert
ENV TERM xterm-256color
COPY . .
RUN pip install poetry
RUN poetry install
CMD ["poetry", "run", "tui"]