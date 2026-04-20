# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install Node.js to build the frontend
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install them
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Build the Vite frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Change back to root directory
WORKDIR /app

# Expose the port Hugging Face expects
EXPOSE 7860

# Run the FastAPI server on port 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
