FROM kalilinux/kali-rolling:latest

# Install real penetration testing tools
RUN apt-get update && apt-get install -y \
    nmap \
    masscan \
    nikto \
    sqlmap \
    metasploit-framework \
    burpsuite \
    gobuster \
    dirb \
    hydra \
    john \
    hashcat \
    aircrack-ng \
    wireshark \
    tcpdump \
    netcat \
    socat \
    curl \
    wget \
    python3 \
    python3-pip \
    git \
    vim \
    tmux \
    screen \
    && rm -rf /var/lib/apt/lists/*

# Install Python exploitation libraries
RUN pip3 install \
    requests \
    beautifulsoup4 \
    selenium \
    scapy \
    paramiko \
    pwntools \
    impacket \
    python-nmap \
    asyncio \
    aiohttp

# Create red team workspace
RUN mkdir -p /redteam/{exploits,payloads,tools,results,intelligence}

# Install custom exploitation frameworks
WORKDIR /redteam/tools
RUN git clone https://github.com/SecureAuthCorp/impacket.git && \
    git clone https://github.com/PowerShellMafia/PowerSploit.git && \
    git clone https://github.com/bettercap/bettercap.git

# Copy red team agent
COPY red_agent.py /redteam/
COPY exploits/ /redteam/exploits/
COPY payloads/ /redteam/payloads/

# Set working directory
WORKDIR /redteam

# Start red team agent
CMD ["python3", "red_agent.py"]