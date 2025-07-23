
# Use the official Python image.
FROM python:3.11-slim

# Set the working directory in the container.
WORKDIR /app

# Copy local code to the container.
COPY . /app

# Install dependencies.
# We assume the orchestrator might need the same core dependencies.
COPY ./packages/xorb_core/requirements.txt /app/packages/xorb_core/requirements.txt
RUN pip install --no-cache-dir -r /app/packages/xorb_core/requirements.txt

# Command to run the orchestrator.
CMD ["python", "-u", "services/orchestrator/main.py"]
