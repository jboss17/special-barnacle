FROM python:3.10-slim

# For CPU version
#FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only code and config files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default so user can run any script manually
CMD ["bash"]