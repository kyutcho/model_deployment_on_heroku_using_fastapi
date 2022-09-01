FROM python:3.8.6 
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}