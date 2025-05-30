# Use an official Freqtrade Docker image as the base
FROM freqtradeorg/freqtrade:stable

# Set the working directory
WORKDIR /freqtrade/user_data

# Copy user-specific files (strategies, configurations, etc.)
# We will add specific COPY commands here in later steps
# For now, this is a placeholder for future modifications

# Expose the default port for FreqUI (optional, as docker-compose will handle port mapping)
EXPOSE 8080

# The default command is defined in docker-compose.yml
# CMD ["trade", "--config", "/freqtrade/user_data/config.json", "--strategy", "ExampleStrategy"]
