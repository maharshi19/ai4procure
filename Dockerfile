FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai4procure.py .
COPY ai4procure_dashboard.html .
COPY Headers_xlsx_Sheet1.csv .
COPY Book2_xlsx_Sheet1.csv .
COPY Book5.xlsx .

ENV PYTHONUNBUFFERED=1
ENV PORT=5000

EXPOSE 5000

CMD gunicorn ai4procure:app \
    --workers 2 \
    --bind 0.0.0.0:$PORT \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
