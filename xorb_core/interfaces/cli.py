"""
Command Line Interface for Xorb Core

Simple CLI for testing and administration of the Xorb platform.
"""

from __future__ import annotations

import asyncio
import json
import sys
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

import click

from ..domain import AgentCapability, TargetId
from .dependencies import get_dependencies

__all__ = ["cli"]


@click.group()
@click.version_option(version="2.0.0", prog_name="xorb")
def cli():
    """Xorb Security Intelligence Platform CLI"""
    pass


@cli.group()
def campaign():
    """Campaign management commands"""
    pass


@campaign.command()
@click.option("--name", required=True, help="Campaign name")
@click.option("--target-id", required=True, help="Target UUID")
@click.option("--max-cost", default="100.00", help="Maximum cost in USD")
@click.option("--max-duration", default=24, help="Maximum duration in hours")
@click.option("--max-api-calls", default=10000, help="Maximum API calls")
@click.option("--capabilities", required=True, help="Comma-separated agent capabilities")
@click.option("--max-agents", default=5, help="Maximum number of agents")
@click.option("--auto-start", is_flag=True, help="Start campaign immediately")
def create(
    name: str,
    target_id: str,
    max_cost: str,
    max_duration: int,
    max_api_calls: int,
    capabilities: str,
    max_agents: int,
    auto_start: bool
):
    """Create a new security campaign"""
    
    async def _create_campaign():
        try:
            # Parse capabilities
            capability_list = [cap.strip() for cap in capabilities.split(",")]
            parsed_capabilities = [AgentCapability(cap) for cap in capability_list]
            
            # Get dependencies
            deps = await get_dependencies()
            campaign_service = deps.get("campaign_service")
            
            # Create campaign
            campaign_id = await campaign_service.create_and_start_campaign(
                name=name,
                target_id=TargetId.from_string(target_id),
                max_cost_usd=Decimal(max_cost),
                max_duration_hours=max_duration,
                max_api_calls=max_api_calls,
                required_capabilities=parsed_capabilities,
                max_agents=max_agents,
                auto_start=auto_start
            )
            
            click.echo(f"Campaign created successfully: {campaign_id}")
            
            # Show progress
            progress = await campaign_service.get_campaign_progress(campaign_id)
            click.echo(json.dumps(progress, indent=2, default=str))
            
        except Exception as e:
            click.echo(f"Error creating campaign: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_create_campaign())


@campaign.command()
@click.argument("campaign_id")
def status(campaign_id: str):
    """Get campaign status"""
    
    async def _get_status():
        try:
            # Get dependencies
            deps = await get_dependencies()
            campaign_service = deps.get("campaign_service")
            
            # Get campaign progress
            from ..domain import CampaignId
            progress = await campaign_service.get_campaign_progress(
                CampaignId.from_string(campaign_id)
            )
            
            click.echo(json.dumps(progress, indent=2, default=str))
            
        except Exception as e:
            click.echo(f"Error getting campaign status: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_get_status())


@cli.group()
def knowledge():
    """Knowledge management commands"""
    pass


@knowledge.command()
@click.option("--text", required=True, help="Text to generate embedding for")
@click.option("--model", default="nvidia/embed-qa-4", help="Embedding model")
@click.option("--input-type", default="query", help="Input type (query/passage)")
def embed(text: str, model: str, input_type: str):
    """Generate text embedding"""
    
    async def _generate_embedding():
        try:
            # Get dependencies
            deps = await get_dependencies()
            knowledge_service = deps.get("knowledge_service")
            
            # Generate embedding
            from ..application import GenerateEmbeddingCommand
            embedding = await knowledge_service._generate_embedding_use_case.execute(
                GenerateEmbeddingCommand(
                    text=text,
                    model=model,
                    input_type=input_type
                )
            )
            
            click.echo(f"Embedding generated successfully:")
            click.echo(f"Dimension: {embedding.dimension}")
            click.echo(f"Model: {embedding.model}")
            click.echo(f"Vector (first 10 values): {list(embedding.vector)[:10]}")
            
        except Exception as e:
            click.echo(f"Error generating embedding: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_generate_embedding())


@cli.group()
def finding():
    """Finding management commands"""
    pass


@finding.command()
@click.argument("finding_id")
def insights(finding_id: str):
    """Get finding insights"""
    
    async def _get_insights():
        try:
            # Get dependencies
            deps = await get_dependencies()
            finding_service = deps.get("finding_service")
            
            # Get insights
            from ..domain import FindingId
            insights = await finding_service.get_finding_insights(
                FindingId.from_string(finding_id)
            )
            
            click.echo(json.dumps(insights, indent=2, default=str))
            
        except Exception as e:
            click.echo(f"Error getting finding insights: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_get_insights())


@cli.command()
def health():
    """Check system health"""
    
    async def _check_health():
        try:
            # Get dependencies
            deps = await get_dependencies()
            
            # Check each service
            services = {
                "campaign_service": "campaign_service",
                "finding_service": "finding_service",
                "knowledge_service": "knowledge_service",
                "embedding_service": "embedding_service",
                "cache_service": "cache_service",
                "notification_service": "notification_service"
            }
            
            health_status = {}
            
            for service_name, key in services.items():
                try:
                    service = deps.get(key)
                    health_status[service_name] = "healthy"
                except Exception as e:
                    health_status[service_name] = f"error: {e}"
            
            # Overall status
            overall_healthy = all(
                status == "healthy" for status in health_status.values()
            )
            
            result = {
                "status": "healthy" if overall_healthy else "degraded",
                "services": health_status,
                "version": "2.0.0"
            }
            
            click.echo(json.dumps(result, indent=2))
            
        except Exception as e:
            click.echo(f"Error checking health: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_check_health())


@cli.command()
@click.option("--host", default="0.0.0.0", help="API server host")
@click.option("--port", default=8000, help="API server port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the API server"""
    
    from .main import run_api_server
    
    click.echo(f"Starting Xorb API server on {host}:{port}")
    asyncio.run(run_api_server(host=host, port=port, reload=reload))


if __name__ == "__main__":
    cli()