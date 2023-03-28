FROM python:3.8.0-slim

# Set the working directory within the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install required packages and dependencies
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

EXPOSE 8000

CMD ["python", "app.py", "runserver", "0.0.0.0:8000"]


