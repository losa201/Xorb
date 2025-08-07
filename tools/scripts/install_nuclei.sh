#!/bin/bash

# This script downloads and installs the latest version of Nuclei.

# Set the Nuclei version.
VERSION="v3.4.7"

# Download the Nuclei binary.
curl -L https://github.com/projectdiscovery/nuclei/releases/download/${VERSION}/nuclei_${VERSION#v}_linux_amd64.zip -o nuclei.zip

# Unzip the Nuclei binary.
unzip -o nuclei.zip

# Move the Nuclei binary to /usr/local/bin.
mv nuclei /usr/local/bin/

# Clean up the downloaded files.
rm nuclei.zip