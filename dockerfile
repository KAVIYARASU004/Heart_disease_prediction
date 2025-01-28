# Use official Python image as a base
FROM python:3.12.5

# Set the working directory in the container
WORKDIR /app

# Copy all files from the current directory to /app in the container
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit default port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
