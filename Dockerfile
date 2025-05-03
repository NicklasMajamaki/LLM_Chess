# Use Miniconda base image
FROM continuumio/miniconda3

# Install system-level deps
RUN apt-get update && apt-get install -y curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# Set environment variables
ENV CONDA_ENV_NAME=llm_chess

# NOTE: For the environment.yml file this should have both conda and pip dependencies
# You'll need to call 'conda env export --from-history > environment.yml'
# Then you need to manually add the pip depdencies to this environment.yml file
# For that, call 'pip freeze > pip-requirements.txt' and copy only the clean dependencies (anything like openai==0.9.1 or build==1.2.2.post1)
COPY environment.yml /tmp/environment.yml

# Create the Conda environment (and clean up afterwards)
RUN conda env create -f /tmp/environment.yml && \
    conda clean -a -y

# Activate conda environment by default
SHELL ["conda", "run", "-n", "llm_chess", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app
COPY . /app