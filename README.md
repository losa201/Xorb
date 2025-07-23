# Xorb 2.0: AI-Powered Security Intelligence Platform

Xorb 2.0 is a complete redesign of the original Xorb and Cetol projects, rebuilt from the ground up as a modern, scalable, and resilient microservices platform. It leverages a state-of-the-art technology stack to provide a robust foundation for building and orchestrating AI-powered security agents.

## Architecture Overview

Xorb 2.0 is built on a service-oriented architecture, with the following key components:

- **API (FastAPI):** A high-performance API server that provides a RESTful interface for interacting with the Xorb platform.
- **Worker (Temporal):** A fleet of Temporal workers that execute long-running, reliable workflows for tasks like reconnaissance, analysis, and reporting.
- **Core Library (xorb-core):** A shared Python package that contains the core business logic, data models, and clients for interacting with external services.
- **Knowledge Fabric:** A combination of PostgreSQL with PGvector for structured and vector data, and Neo4j for graph-based intelligence.
- **Frontend (Next.js):** A modern, responsive web interface for visualizing data, managing workflows, and viewing reports (coming soon).

## Getting Started (Local Development)

To get started with Xorb 2.0, you'll need Docker and Docker Compose installed.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/losa201/Xorb.git
    cd Xorb
    ```

2.  **Create your environment file:**

    ```bash
    cp .env.example .env
    ```

    Now, edit the `.env` file and add your HackerOne API credentials.

3.  **Build and run the services:**

    ```bash
    docker-compose up --build
    ```

4.  **Access the API:**

    The API server will be running at `http://localhost:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

5.  **Access the Temporal UI:**

    The Temporal web UI will be running at `http://localhost:8233`.

## Development Guide

This project is structured as a monorepo. The core logic is in `packages/xorb_core`, and the services are in the `services` directory.

-   **To add a new client or model:** Add it to the `xorb_core` package.
-   **To add a new workflow:** Add a new workflow definition in `services/worker/app/workflows.py` and a corresponding activity in `services/worker/app/activities.py`.
-   **To add a new API endpoint:** Add a new router in `services/api/app/routers/`.

## Next Steps

Now that the foundational architecture is in place, the next steps are to:

1.  **Port the rest of the Cetol and Xorb logic:** Systematically move the agent framework, knowledge base, and other key features from the original projects into the new `xorb_core` library.
2.  **Build out the API:** Create a comprehensive set of API endpoints for managing targets, triggering workflows, and retrieving results.
3.  **Develop the Frontend:** Build the Next.js dashboard to provide a rich user experience for interacting with the platform.
4.  **Implement the Knowledge Fabric:** Flesh out the data models and create the necessary logic for interacting with PostgreSQL/PGvector and Neo4j.
