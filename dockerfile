# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code and requirements file to the container
COPY . .

# Upgrade pip and install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r req.txt

# Expose the port FastAPI will run on
EXPOSE 8001

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
