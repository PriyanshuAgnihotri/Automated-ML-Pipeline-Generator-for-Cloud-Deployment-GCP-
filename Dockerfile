FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip install -e ".[dev]"
RUN python -m unittest discover -s tests/unit
