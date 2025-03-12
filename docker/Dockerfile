FROM ubuntu:22.04

# Set some default environment variables plus PYTHON_VERSION, REPO, TAG and WORKSPACE
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=Etc/UTC \
    PYTHON_VERSION=3.12 \
    REPO=freqtrade/freqtrade \
    TAG=2025.2 \
    WORKSPACE=/freqtrade

# Prevent interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies including ta-lib
RUN apt-get update && apt-get install -y build-essential \
    libcurl4-openssl-dev \
    libffi-dev \
    tk-dev \
    xz-utils \
    curl \
    lsb-release \
    git \
    wget \
    libncursesw5-dev \
    zlib1g-dev \
    libpq-dev \
    libmysqlclient-dev \
    pkg-config \
    libssl-dev \
    unixodbc-dev \
    locales \
    locales-all \
    tzdata \
    python3.10-venv && rm -rf /var/lib/apt/lists/*

# Configure locale
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# Install uv (Uvicorn) for the freqtrade environment
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/bin" sh

# Create a non-root user for better security
RUN useradd -ms /bin/bash runner

# Create workspace directory and set permissions
RUN mkdir -p ${WORKSPACE} && \
    chown -R runner:runner ${WORKSPACE} && \
    chmod -R u+rwX,go+rX,go-w ${WORKSPACE}

# Switch to the non-root user and set the working directory
USER runner
WORKDIR $WORKSPACE

# Install Python version
RUN uv python install -v ${PYTHON_VERSION}

# Checkout the project at the given tag
RUN git clone "https://github.com/${REPO}.git" "${WORKSPACE}" && \
    cd "${WORKSPACE}" && \
    git checkout "${TAG}"

RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.6.3/ta-lib-0.6.3-src.tar.gz -O /tmp/ta-lib.tar.gz

# List contents of /tmp to see if the tarball was downloaded
RUN ls -l /tmp

# Extract the tarball and list contents of /tmp to verify
RUN tar -zxvf /tmp/ta-lib.tar.gz -C /tmp
RUN ls -l /tmp

# Verify the extraction by checking contents of the directory
RUN ls -l /tmp/ta-lib-0.6.3

# Navigate to the TA-Lib directory and configure
USER root
WORKDIR /tmp/ta-lib-0.6.3
RUN ./configure && \
    make && \
    make install && \
    rm -rf /tmp/ta-lib*


RUN python3 -m venv .venv
# Install other dependencies in the virtual environment
RUN  ./.venv/bin/python3 -m pip install --no-cache-dir setuptools
RUN  ./.venv/bin/python3 -m pip install --no-cache-dir numpy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.10-dev \
    python3-pip \
    curl \
    && apt-get clean

# Download and install TA-Lib from source
RUN curl -L https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -o ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz


# Ensure pip is up to date
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Install TA-Lib with no build isolation
RUN  ./.venv/bin/python3 -m pip install TA-Lib

WORKDIR $WORKSPACE
RUN python3 -m venv .venv

# Install additional dependencies
RUN  ./.venv/bin/python3 -m pip install -r requirements-dev.txt

RUN uv run python -m pytest -v tests || true

# CMD to run tests
CMD ["uv", "run", "–offline", "python", "-m", "pytest", "-vv", "-rA", "--tb=short", "tests"]
