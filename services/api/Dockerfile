# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy the project files into the container
COPY pyproject.toml poetry.lock .

# Install project dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-dev

# Copy the application code into the container
COPY . .

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
