# Use the official Python image as the base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the /app directory in the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]
