FROM python:3.10
WORKDIR /ToT
RUN pip install openai==0.27.7
RUN pip install hugchat
RUN pip install bardapi
