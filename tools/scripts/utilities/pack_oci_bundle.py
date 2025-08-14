#!/usr/bin/env python3
"""
XORB OCI Bundle Creator
Creates offline deployment bundles for airgapped environments
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import docker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XorbOCIBundler:
    def __init__(self, output_dir="deployment", include_data=False):
        self.output_dir = Path(output_dir)
        self.include_data = include_data
        self.bundle_name = f"xorb_oci_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.bundle_path = self.output_dir / f"{self.bundle_name}.tar.gz"
        self.temp_dir = Path("/tmp") / self.bundle_name
        self.docker_client = docker.from_env()

        # XORB images to include
        self.xorb_images = [
            "postgres:15-alpine",
            "redis:7-alpine",
            "neo4j:5-community",
            "qdrant/qdrant:latest",
            "prom/prometheus:latest",
            "grafana/grafana:latest",
            "prom/node-exporter:latest"
        ]

        # XORB service images (to be built)
        self.xorb_services = [
            "xorb-api",
            "xorb-orchestrator",
            "xorb-worker"
        ]

    def detect_architecture(self):
        """Detect system architecture"""
        arch_map = {
            "x86_64": "amd64",
            "aarch64": "arm64",
            "armv7l": "arm"
        }

        uname_arch = subprocess.check_output(["uname", "-m"]).decode().strip()
        return arch_map.get(uname_arch, uname_arch)

    def create_bundle_structure(self):
        """Create bundle directory structure"""
        logger.info(f"Creating bundle structure in {self.temp_dir}")

        # Create directory structure
        directories = [
            "images",
            "config",
            "scripts",
            "docs",
            "systemd",
            "data"
        ]

        for directory in directories:
            (self.temp_dir / directory).mkdir(parents=True, exist_ok=True)

        logger.info("Bundle structure created")

    def export_docker_images(self):
        """Export all required Docker images"""
        logger.info("Exporting Docker images...")

        all_images = self.xorb_images + self.xorb_services
        images_dir = self.temp_dir / "images"

        # Create images manifest
        manifest = {
            "format_version": "1.0",
            "created": datetime.now().isoformat(),
            "architecture": self.detect_architecture(),
            "images": []
        }

        for image in all_images:
            try:
                logger.info(f"Exporting image: {image}")

                # Pull/build image if needed
                if image in self.xorb_services:
                    # Build XORB service image
                    self._build_xorb_service(image)
                else:
                    # Pull external image
                    self.docker_client.images.pull(image)

                # Export image to tar
                docker_image = self.docker_client.images.get(image)
                image_filename = f"{image.replace(':', '_').replace('/', '_')}.tar"
                image_path = images_dir / image_filename

                with open(image_path, 'wb') as f:
                    for chunk in docker_image.save():
                        f.write(chunk)

                # Add to manifest
                manifest["images"].append({
                    "name": image,
                    "filename": image_filename,
                    "size": image_path.stat().st_size,
                    "id": docker_image.id
                })

                logger.info(f"Exported {image} -> {image_filename}")

            except Exception as e:
                logger.error(f"Failed to export {image}: {e}")

        # Save manifest
        with open(images_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Exported {len(manifest['images'])} Docker images")

    def _build_xorb_service(self, service_name):
        """Build XORB service image"""
        dockerfile_map = {
            "xorb-api": "services/api/Dockerfile",
            "xorb-orchestrator": "services/orchestrator/Dockerfile",
            "xorb-worker": "services/worker/Dockerfile"
        }

        dockerfile_path = dockerfile_map.get(service_name)
        if not dockerfile_path or not Path(dockerfile_path).exists():
            logger.warning(f"Dockerfile not found for {service_name}, skipping")
            return

        logger.info(f"Building {service_name}...")

        try:
            # Build image
            image, logs = self.docker_client.images.build(
                path=".",
                dockerfile=dockerfile_path,
                tag=service_name,
                buildargs={"ARCH": self.detect_architecture()}
            )

            logger.info(f"Built {service_name} successfully")

        except Exception as e:
            logger.error(f"Failed to build {service_name}: {e}")

    def copy_configuration(self):
        """Copy configuration files"""
        logger.info("Copying configuration files...")

        config_dir = self.temp_dir / "config"

        # Copy configuration files
        config_files = [
            "docker-compose.local.yml",
            "docker-compose.complete.yml",
            "config/local",
            ".env.example"
        ]

        for config_file in config_files:
            src_path = Path(config_file)
            if src_path.exists():
                if src_path.is_dir():
                    shutil.copytree(src_path, config_dir / src_path.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, config_dir)
                logger.info(f"Copied {config_file}")
            else:
                logger.warning(f"Configuration file not found: {config_file}")

    def copy_scripts(self):
        """Copy installation and management scripts"""
        logger.info("Copying scripts...")

        scripts_dir = self.temp_dir / "scripts"

        # Copy essential scripts
        script_files = [
            "autodeploy.sh",
            "scripts/monitor",
            "scripts/security",
            "scripts/test"
        ]

        for script_file in script_files:
            src_path = Path(script_file)
            if src_path.exists():
                if src_path.is_dir():
                    shutil.copytree(src_path, scripts_dir / src_path.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, scripts_dir)
                    # Make scripts executable
                    os.chmod(scripts_dir / src_path.name, 0o755)
                logger.info(f"Copied {script_file}")
            else:
                logger.warning(f"Script not found: {script_file}")

    def copy_systemd_files(self):
        """Copy systemd service files"""
        logger.info("Copying systemd files...")

        systemd_dir = self.temp_dir / "systemd"
        src_systemd = Path("systemd")

        if src_systemd.exists():
            shutil.copytree(src_systemd, systemd_dir, dirs_exist_ok=True)
            logger.info("Copied systemd service files")
        else:
            logger.warning("Systemd directory not found")

    def copy_documentation(self):
        """Copy documentation"""
        logger.info("Copying documentation...")

        docs_dir = self.temp_dir / "docs"

        # Copy documentation files
        doc_files = [
            "README.md",
            "DEPLOYMENT_GUIDE.md",
            "CLAUDE.md",
            "docs"
        ]

        for doc_file in doc_files:
            src_path = Path(doc_file)
            if src_path.exists():
                if src_path.is_dir():
                    shutil.copytree(src_path, docs_dir / src_path.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, docs_dir)
                logger.info(f"Copied {doc_file}")

    def copy_data(self):
        """Copy sample/seed data if requested"""
        if not self.include_data:
            return

        logger.info("Copying data files...")

        data_dir = self.temp_dir / "data"

        # Copy data files
        data_files = [
            "data",
            "seeds",
            "migrations"
        ]

        for data_file in data_files:
            src_path = Path(data_file)
            if src_path.exists():
                if src_path.is_dir():
                    shutil.copytree(src_path, data_dir / src_path.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, data_dir)
                logger.info(f"Copied {data_file}")

    def create_installer_script(self):
        """Create offline installer script"""
        logger.info("Creating installer script...")

        installer_script = self.temp_dir / "install_xorb_offline.sh"

        with open(installer_script, 'w') as f:
            f.write('''#!/bin/bash

# XORB Offline Installer
# Installs XORB from OCI bundle in airgapped environments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XORB_ROOT="${SCRIPT_DIR}"

# Colors
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
RED='\\033[0;31m'
NC='\\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check for Docker
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi

    # Check for Docker Compose
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "Docker Compose not found. Please install Docker Compose first."
        exit 1
    fi

    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon not running. Please start Docker."
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Load Docker images
load_images() {
    log_info "Loading Docker images..."

    local images_dir="${XORB_ROOT}/images"
    local manifest_file="${images_dir}/manifest.json"

    if [[ ! -f "${manifest_file}" ]]; then
        log_error "Images manifest not found: ${manifest_file}"
        exit 1
    fi

    # Read manifest and load images
    while IFS= read -r image_file; do
        local image_path="${images_dir}/${image_file}"
        if [[ -f "${image_path}" ]]; then
            log_info "Loading image: ${image_file}"
            docker load -i "${image_path}"
        else
            log_warning "Image file not found: ${image_file}"
        fi
    done < <(jq -r '.images[].filename' "${manifest_file}")

    log_info "Images loaded successfully"
}

# Install configuration
install_config() {
    log_info "Installing configuration..."

    # Copy configuration files to appropriate locations
    cp -r "${XORB_ROOT}/config/"* ./

    # Make scripts executable
    find "${XORB_ROOT}/scripts" -name "*.sh" -exec chmod +x {} \\;
    find "${XORB_ROOT}/scripts" -name "*.py" -exec chmod +x {} \\;

    log_info "Configuration installed"
}

# Install systemd services
install_systemd() {
    log_info "Installing systemd services..."

    if [[ -d "${XORB_ROOT}/systemd" ]]; then
        sudo cp "${XORB_ROOT}/systemd/"*.service /etc/systemd/system/
        sudo cp "${XORB_ROOT}/systemd/"*.target /etc/systemd/system/
        sudo systemctl daemon-reload
        log_info "Systemd services installed"
    else
        log_warning "Systemd directory not found, skipping"
    fi
}

# Deploy XORB
deploy_xorb() {
    log_info "Deploying XORB..."

    # Use local docker-compose file
    if [[ -f "docker-compose.local.yml" ]]; then
        docker-compose -f docker-compose.local.yml up -d
    else
        log_error "Docker Compose file not found"
        exit 1
    fi

    log_info "XORB deployment started"
}

# Main installation function
main() {
    echo "XORB Offline Installer"
    echo "======================"
    echo

    check_prerequisites
    load_images
    install_config
    install_systemd
    deploy_xorb

    echo
    echo -e "${GREEN}🎉 XORB installation completed!${NC}"
    echo
    echo "Access points:"
    echo "  API:      http://localhost:8000"
    echo "  Grafana:  http://localhost:3000"
    echo "  Prometheus: http://localhost:9090"
    echo
    echo "Management:"
    echo "  Status:   docker-compose -f docker-compose.local.yml ps"
    echo "  Logs:     docker-compose -f docker-compose.local.yml logs -f"
    echo "  Stop:     docker-compose -f docker-compose.local.yml down"
    echo
}

main "$@"
''')

        # Make installer executable
        os.chmod(installer_script, 0o755)
        logger.info("Installer script created")

    def create_bundle_metadata(self):
        """Create bundle metadata file"""
        logger.info("Creating bundle metadata...")

        metadata = {
            "bundle_name": self.bundle_name,
            "version": "2.0.0",
            "created": datetime.now().isoformat(),
            "architecture": self.detect_architecture(),
            "format_version": "1.0",
            "description": "XORB Autonomous Security Platform - Offline Deployment Bundle",
            "components": {
                "docker_images": len(self.xorb_images + self.xorb_services),
                "configuration": True,
                "scripts": True,
                "documentation": True,
                "systemd_services": True,
                "sample_data": self.include_data
            },
            "requirements": {
                "docker": ">=20.0.0",
                "docker_compose": ">=1.28.0",
                "min_memory_gb": 4,
                "min_disk_gb": 20,
                "supported_architectures": ["amd64", "arm64"]
            },
            "installation": {
                "installer_script": "install_xorb_offline.sh",
                "estimated_time_minutes": 10,
                "requires_internet": False
            }
        }

        with open(self.temp_dir / "bundle_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create README for the bundle
        readme_content = f"""# XORB Offline Deployment Bundle

This bundle contains everything needed to deploy XORB in an offline/airgapped environment.

## Bundle Information
- Bundle: {self.bundle_name}
- Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Architecture: {self.detect_architecture()}
- Version: 2.0.0

## Contents
- Docker images for all XORB services
- Configuration templates
- Installation scripts
- Documentation
- Systemd service files

## Installation

1. Extract the bundle:
   ```bash
   tar -xzf {self.bundle_name}.tar.gz
   cd {self.bundle_name}
   ```

2. Run the installer:
   ```bash
   ./install_xorb_offline.sh
   ```

3. Verify installation:
   ```bash
   docker-compose -f docker-compose.local.yml ps
   ```

## Requirements
- Docker >= 20.0.0
- Docker Compose >= 1.28.0
- Minimum 4GB RAM
- Minimum 20GB disk space
- Linux x86_64 or ARM64

## Support
Refer to the documentation in the `docs/` directory for detailed installation and configuration instructions.
"""

        with open(self.temp_dir / "README.txt", 'w') as f:
            f.write(readme_content)

        logger.info("Bundle metadata created")

    def create_tar_bundle(self):
        """Create final tar.gz bundle"""
        logger.info(f"Creating bundle archive: {self.bundle_path}")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create tar.gz bundle
        with tarfile.open(self.bundle_path, 'w:gz') as tar:
            tar.add(self.temp_dir, arcname=self.bundle_name)

        # Get bundle size
        bundle_size_mb = self.bundle_path.stat().st_size / 1024 / 1024

        logger.info(f"Bundle created: {self.bundle_path} ({bundle_size_mb:.1f}MB)")

        # Create checksum
        checksum_file = self.bundle_path.with_suffix('.sha256')
        subprocess.run(['sha256sum', str(self.bundle_path)], stdout=open(checksum_file, 'w'))

        logger.info(f"Checksum created: {checksum_file}")

    def cleanup(self):
        """Clean up temporary files"""
        logger.info("Cleaning up temporary files...")

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        logger.info("Cleanup completed")

    def create_bundle(self):
        """Create complete OCI bundle"""
        logger.info(f"Creating XORB OCI bundle: {self.bundle_name}")

        try:
            self.create_bundle_structure()
            self.export_docker_images()
            self.copy_configuration()
            self.copy_scripts()
            self.copy_systemd_files()
            self.copy_documentation()
            self.copy_data()
            self.create_installer_script()
            self.create_bundle_metadata()
            self.create_tar_bundle()

            logger.info("Bundle creation completed successfully!")

            return {
                "bundle_path": str(self.bundle_path),
                "bundle_size_mb": self.bundle_path.stat().st_size / 1024 / 1024,
                "checksum_file": str(self.bundle_path.with_suffix('.sha256'))
            }

        except Exception as e:
            logger.error(f"Bundle creation failed: {e}")
            raise
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description="Create XORB OCI deployment bundle")
    parser.add_argument("--output-dir", default="deployment",
                       help="Output directory for bundle (default: deployment)")
    parser.add_argument("--include-data", action="store_true",
                       help="Include sample/seed data in bundle")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        bundler = XorbOCIBundler(args.output_dir, args.include_data)
        result = bundler.create_bundle()

        print("\\n🎉 XORB OCI Bundle created successfully!")
        print(f"Bundle: {result['bundle_path']}")
        print(f"Size: {result['bundle_size_mb']:.1f}MB")
        print(f"Checksum: {result['checksum_file']}")
        print("\\nTo deploy in airgapped environment:")
        print("1. Transfer bundle to target system")
        print(f"2. Extract: tar -xzf {Path(result['bundle_path']).name}")
        print("3. Install: ./install_xorb_offline.sh")

    except Exception as e:
        logger.error(f"Failed to create bundle: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
