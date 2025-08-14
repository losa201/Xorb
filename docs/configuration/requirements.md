# Requirements Configuration

This directory contains multiple requirements files for different components of the XORB system:

## requirements.txt
Core dependencies for the base XORB system:
- Python 3.10+
- asyncio
- logging
- pathlib
- datetime
- json
- typing

## requirements-execution.txt
Dependencies for execution components:
- docker
- kubernetes
- prometheus-client
- grafana-api
- pydantic
- fastapi

## requirements-ml.txt
Machine learning dependencies for analytical components:
- numpy
- pandas
- scikit-learn
- tensorflow
- torch
- xgboost

Each requirements file serves a specific purpose in the XORB architecture:
- `requirements.txt` contains base dependencies needed for core functionality
- `requirements-execution.txt` includes dependencies for execution and orchestration components
- `requirements-ml.txt` provides dependencies for machine learning and analytical capabilities

The requirements files are designed to support modular deployment based on the specific needs of each component.
