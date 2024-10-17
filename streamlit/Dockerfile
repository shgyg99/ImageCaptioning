FROM python:3.9-slim
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /app

COPY . .



EXPOSE 3000

CMD ["streamlit", "run", "app.py"]