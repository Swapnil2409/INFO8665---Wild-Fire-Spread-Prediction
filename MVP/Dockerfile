FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt

EXPOSE 5000

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "api/app.py"]
