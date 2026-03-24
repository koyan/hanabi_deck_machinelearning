FROM tensorflow/tensorflow:2.15.0

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We don't COPY the script anymore! 
# We just tell Docker to run whatever is named 'train_hanabi.py' in /app
CMD ["python", "train_hanabi.py"]