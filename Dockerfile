FROM python:3.11-slim-buster

WORKDIR /sfm
COPY . /sfm

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download ru_core_news_lg
RUN python -m nltk.downloader punkt

CMD ["python3", "bot_starter.py"]