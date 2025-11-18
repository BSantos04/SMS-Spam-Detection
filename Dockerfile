# Use Snakemake as base image
FROM python:3.12-slim

# Create working directory
WORKDIR /spam-detect
COPY SMS_SPAM_Detection.py /spam-detect/SMS_SPAM_Detection.py
COPY datasets/ /spam-detect/datasets/
COPY requirements.txt /spam-detect/requirements.txt

# Install requirements using pip
RUN pip install --no-cache-dir -r requirements.txt

# Prepare the command-line
ENTRYPOINT ["python3", "/spam-detect/SMS_SPAM_Detection.py"]