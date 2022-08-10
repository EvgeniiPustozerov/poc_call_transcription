FROM python:3.10-slim
WORKDIR /app
COPY . .

RUN apt-get -y update
RUN apt-get -y upgrade

# Install every package one after another to track time
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "./Interface.py"]
# Next commands are: docker build -t pustozerov/poc-call-transcription:1.0
