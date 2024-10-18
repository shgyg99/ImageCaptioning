FROM python:3.9-slim

RUN pip install --upgrade pip
RUN pip install torch
RUN  pip install torchvision==0.17.1
RUN pip install torchtext==0.17.1
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .
EXPOSE 3000
CMD ["streamlit", "run", "app.py"]