# Use Python 3.10 slim base image for a smaller footprint
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for some Python packages (like xgboost or scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create the model directory if it doesn't exist (though it should be copied if local)
RUN mkdir -p model

# Expose the port specified in requirements
EXPOSE 8003

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
