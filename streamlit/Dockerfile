FROM python:3.9-slim
RUN pip install --upgrade pip
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 3000

CMD ["streamlit", "run", "app.py"]