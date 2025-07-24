# XORB 2.0 Enhanced on a Single VPS

This document outlines the steps to install, upgrade, and manage the XORB 2.0 Enhanced application on a single-node k3s cluster.

## Prerequisites

* A single VPS with 8 vCPU, 16 GB RAM, and 200 GB NVMe.
* k3s 1.30.x installed with Traefik ingress enabled.
* A public domain pointing to the VPS IP address.
* `kubectl` and `helm` installed on your local machine.
* Flux v2 installed on the cluster.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create secrets:**

    Create a `secrets.yaml` file with the following content:

    ```yaml
    apiVersion: v1
    kind: Secret
    metadata:
      name: xorb-api-secrets
    stringData:
      POSTGRES_PASSWORD: <your-postgres-password>
      NEO4J_AUTH: <your-neo4j-auth>
      JWT_SECRET: <your-jwt-secret>
    ---
    apiVersion: v1
    kind: Secret
    metadata:
      name: xorb-worker-secrets
    stringData:
      POSTGRES_PASSWORD: <your-postgres-password>
      NEO4J_AUTH: <your-neo4j-auth>
    ```

    Apply the secrets to the cluster:

    ```bash
    kubectl apply -f secrets.yaml
    ```

3.  **Deploy the application:**

    The GitOps flow will automatically deploy the application. To trigger the first deployment, commit and push a change to the repository.

## Upgrade

To upgrade the application, simply push a new commit to the `main` branch. The CI/CD pipeline will build new container images, update the image tags in the `HelmRelease`, and Flux will automatically roll out the new version.

## Rollback

To roll back to a previous version, revert the commit that introduced the new image tags and push the change to the `main` branch. Flux will detect the change and roll back to the previous version.

## Troubleshooting

*   **Check pod status:**

    ```bash
    kubectl get pods -n default
    ```

*   **Check pod logs:**

    ```bash
    kubectl logs -n default <pod-name>
    ```

*   **Check Flux reconciliation:**

    ```bash
    flux get kustomizations
    flux get helmreleases
    ```
