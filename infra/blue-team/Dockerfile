FROM ubuntu:22.04

# Install security monitoring and defensive tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    tcpdump \
    wireshark-common \
    tshark \
    suricata \
    fail2ban \
    aide \
    rkhunter \
    chkrootkit \
    clamav \
    clamav-daemon \
    osquery \
    htop \
    iftop \
    nethogs \
    iotop \
    sysstat \
    auditd \
    rsyslog \
    logrotate \
    curl \
    wget \
    git \
    vim \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python security libraries
RUN pip3 install \
    scapy \
    psutil \
    requests \
    asyncio \
    aiohttp \
    watchdog \
    yara-python \
    volatility3 \
    pandas \
    numpy \
    scikit-learn \
    elasticsearch \
    loguru

# Install threat intelligence tools
RUN pip3 install \
    misp-python \
    cortex4py \
    thehive4py \
    shodan \
    virustotal-api

# Create blue team workspace
RUN mkdir -p /blueteam/{rules,logs,reports,intelligence,models}

# Copy blue team agent and configurations
COPY blue_agent.py /blueteam/
COPY rules/ /blueteam/rules/
COPY configs/ /blueteam/configs/

# Configure Suricata
COPY suricata.yaml /etc/suricata/
RUN suricata-update

# Configure fail2ban
COPY jail.local /etc/fail2ban/

# Set working directory
WORKDIR /blueteam

# Start blue team agent
CMD ["python3", "blue_agent.py"]
