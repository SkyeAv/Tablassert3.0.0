FROM python:3.13

WORKDIR /user/local/tablassert

ENV TERM=xterm-256color
ENV COLORTERM=truecolor
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

COPY . .

RUN pip install poetry
RUN poetry install

CMD ["poetry", "run", "tui"]