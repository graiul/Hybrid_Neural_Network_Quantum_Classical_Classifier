FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY menu.sh /app/
COPY ./QCC_colored_points.py /app/Colored_points_hybrid_classifier/
# Make menu.sh executable
RUN chmod +x /app/menu.sh
# Set menu.sh as the entry point
ENTRYPOINT ["/app/menu.sh"]



