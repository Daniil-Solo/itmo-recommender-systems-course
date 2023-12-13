FROM python:3.10
WORKDIR /usr/app
ADD requirements.txt .
RUN pip install -r requirements.txt