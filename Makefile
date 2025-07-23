SHELL := /bin/bash

.PHONY: help build test deploy logs

help:
	@echo "Commands:"
	@echo "  build         : Build the Docker images"
	@echo "  test          : Run pytest tests"
	@echo "  deploy        : Deploy the application to Kubernetes"
	@echo "  logs          : Tail the logs of the API deployment"

build:
	docker-compose build

test:
	pytest tests/

deploy:
	helm upgrade --install xorb-api ./helm/xorb-api --namespace xorb --create-namespace -f ./helm/xorb-api/values.yaml

logs:
	kubectl logs -f deployment/xorb-api -n xorb