"""
Interface Layer Entry Point

Main module for starting the REST API and gRPC services.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Optional

import uvicorn
from fastapi import FastAPI

from .rest.app import create_app

__all__ = ["main", "run_api_server"]

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure structured logging"""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("xorb_api.log")
        ]
    )


async def run_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False
) -> None:
    """Run the FastAPI server"""
    
    logger.info(f"Starting Xorb API server on {host}:{port}")
    
    # Create FastAPI app
    app = create_app()
    
    # Configure Uvicorn
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
        access_log=True,
        use_colors=True
    )
    
    # Create and run server
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        logger.info("API server stopped")


async def run_grpc_server(
    host: str = "0.0.0.0",
    port: int = 50051
) -> None:
    """Run the gRPC server"""
    
    logger.info(f"Starting Xorb gRPC server on {host}:{port}")
    
    try:
        import grpc
        from concurrent import futures
        
        # Create gRPC server
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10)
        )
        
        # Add services
        from .grpc.services import create_embedding_service, create_campaign_service
        
        embedding_service = await create_embedding_service()
        campaign_service = await create_campaign_service()
        
        # In a real implementation, you would add the services to the server here
        # server.add_service(EmbeddingServicer_pb2_grpc.add_EmbeddingServiceServicer_to_server,
        #                   embedding_service, server)
        
        # Listen on port
        listen_addr = f"{host}:{port}"
        server.add_insecure_port(listen_addr)
        
        # Start server
        await server.start()
        logger.info(f"gRPC server started on {listen_addr}")
        
        # Wait for termination
        await server.wait_for_termination()
        
    except ImportError:
        logger.warning("gRPC not available, skipping gRPC server")
    except Exception as e:
        logger.error(f"gRPC server failed: {e}")
    finally:
        logger.info("gRPC server stopped")


async def run_servers() -> None:
    """Run both API and gRPC servers"""
    
    # Setup graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        shutdown_event.set()
    
    # Register signal handlers
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())
    
    # Start servers concurrently
    api_task = asyncio.create_task(run_api_server())
    grpc_task = asyncio.create_task(run_grpc_server())
    
    try:
        # Wait for shutdown signal or server completion
        done, pending = await asyncio.wait(
            [api_task, grpc_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    except Exception as e:
        logger.error(f"Server error: {e}")
    
    finally:
        logger.info("All servers stopped")


def main() -> None:
    """Main entry point"""
    
    setup_logging()
    
    logger.info("Starting Xorb Interface Layer")
    
    try:
        asyncio.run(run_servers())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()