#!/usr/bin/env python3
"""
XORB Repository Cleanup Script
Clean up evolution state files and optimize repository structure
"""

import os
import shutil
import gzip
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import tarfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RepositoryCleanup:
    """Repository cleanup and optimization utility"""
    
    def __init__(self, repo_root: str = "/root/Xorb"):
        self.repo_root = Path(repo_root)
        self.backup_dir = self.repo_root / "backups" / "cleanup"
        self.archive_dir = self.repo_root / "data" / "archives"
        
        # Statistics
        self.stats = {
            'evolution_files_found': 0,
            'evolution_files_archived': 0,
            'bytes_saved': 0,
            'directories_created': 0,
            'files_moved': 0
        }
        
    def run_cleanup(self) -> Dict[str, Any]:
        """Run comprehensive repository cleanup"""
        logger.info("🧹 Starting repository cleanup...")
        
        # Create necessary directories
        self._create_directories()
        
        # Clean up evolution state files
        self._cleanup_evolution_files()
        
        # Optimize directory structure
        self._optimize_structure()
        
        # Clean up temporary files
        self._cleanup_temp_files()
        
        # Generate cleanup report
        report = self._generate_report()
        
        logger.info("✅ Repository cleanup completed!")
        return report
        
    def _create_directories(self):
        """Create necessary directories for cleanup"""
        directories = [
            self.backup_dir,
            self.archive_dir,
            self.repo_root / "data" / "evolution_states",
            self.repo_root / "data" / "runtime",
            self.repo_root / "logs" / "archive",
            self.repo_root / "temp"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.stats['directories_created'] += 1
            
        logger.info(f"📁 Created {len(directories)} directory structures")
        
    def _cleanup_evolution_files(self):
        """Clean up evolution state files"""
        logger.info("🔄 Processing evolution state files...")
        
        # Find all evolution state files
        evolution_files = list(self.repo_root.glob("evolution_state_*.json"))
        self.stats['evolution_files_found'] = len(evolution_files)
        
        if not evolution_files:
            logger.info("No evolution state files found")
            return
            
        logger.info(f"Found {len(evolution_files)} evolution state files")
        
        # Group files by date
        file_groups = self._group_files_by_date(evolution_files)
        
        # Archive files by date
        for date, files in file_groups.items():
            self._archive_files_by_date(date, files)
            
        logger.info(f"📦 Archived {self.stats['evolution_files_archived']} evolution files")
        
    def _group_files_by_date(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group evolution files by date"""
        groups = {}
        
        for file in files:
            try:
                # Extract date from filename: evolution_state_20250802_155346.json
                filename = file.name
                date_part = filename.split('_')[2]  # Get 20250802
                date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                
                if date_str not in groups:
                    groups[date_str] = []
                groups[date_str].append(file)
                
            except (IndexError, ValueError) as e:
                logger.warning(f"Failed to parse date from {filename}: {e}")
                # Put in 'unknown' group
                if 'unknown' not in groups:
                    groups['unknown'] = []
                groups['unknown'].append(file)
                
        return groups
        
    def _archive_files_by_date(self, date: str, files: List[Path]):
        """Archive files for a specific date"""
        if not files:
            return
            
        archive_file = self.archive_dir / f"evolution_states_{date}.tar.gz"
        
        logger.info(f"📦 Archiving {len(files)} files for {date}")
        
        with tarfile.open(archive_file, 'w:gz') as tar:
            for file in files:
                try:
                    # Add file to archive with relative path
                    tar.add(file, arcname=file.name)
                    
                    # Track file size before deletion
                    self.stats['bytes_saved'] += file.stat().st_size
                    
                    # Remove original file
                    file.unlink()
                    self.stats['evolution_files_archived'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to archive {file}: {e}")
                    
        logger.info(f"✅ Created archive: {archive_file}")
        
    def _optimize_structure(self):
        """Optimize repository directory structure"""
        logger.info("🏗️ Optimizing repository structure...")
        
        # Move misplaced files to appropriate locations
        moves = [
            # Move logs to logs directory
            (self.repo_root.glob("*.log"), self.repo_root / "logs"),
            # Move config files to config directory
            (self.repo_root.glob("*.yml"), self.repo_root / "config"),
            (self.repo_root.glob("*.yaml"), self.repo_root / "config"),
            # Move JSON data files to data directory
            (self.repo_root.glob("*_results.json"), self.repo_root / "data"),
            (self.repo_root.glob("*_report.json"), self.repo_root / "data"),
        ]
        
        for file_pattern, target_dir in moves:
            for file in file_pattern:
                if file.is_file() and not file.name.startswith('.'):
                    try:
                        target_file = target_dir / file.name
                        if not target_file.exists():
                            shutil.move(str(file), str(target_file))
                            self.stats['files_moved'] += 1
                            logger.debug(f"Moved {file.name} to {target_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to move {file}: {e}")
                        
    def _cleanup_temp_files(self):
        """Clean up temporary and cache files"""
        logger.info("🗑️ Cleaning up temporary files...")
        
        # Patterns for temporary files
        temp_patterns = [
            "*.tmp",
            "*.temp", 
            ".DS_Store",
            "Thumbs.db",
            "*.pyc",
            "__pycache__/*"
        ]
        
        # Cache directories to clean
        cache_dirs = [
            self.repo_root / "__pycache__",
            self.repo_root / ".pytest_cache",
            self.repo_root / ".ruff_cache",
            self.repo_root / "node_modules" / ".cache"
        ]
        
        temp_files_removed = 0
        
        # Remove temporary files
        for pattern in temp_patterns:
            for file in self.repo_root.rglob(pattern):
                if file.is_file():
                    try:
                        file.unlink()
                        temp_files_removed += 1
                    except Exception as e:
                        logger.debug(f"Failed to remove temp file {file}: {e}")
                        
        # Clean cache directories
        for cache_dir in cache_dirs:
            if cache_dir.exists() and cache_dir.is_dir():
                try:
                    shutil.rmtree(cache_dir)
                    temp_files_removed += 1
                except Exception as e:
                    logger.debug(f"Failed to remove cache dir {cache_dir}: {e}")
                    
        logger.info(f"🗑️ Removed {temp_files_removed} temporary files")
        
    def _generate_report(self) -> Dict[str, Any]:
        """Generate cleanup report"""
        report = {
            'cleanup_timestamp': datetime.utcnow().isoformat(),
            'repository_root': str(self.repo_root),
            'statistics': self.stats,
            'space_saved_mb': round(self.stats['bytes_saved'] / (1024 * 1024), 2),
            'archives_created': len(list(self.archive_dir.glob("*.tar.gz"))),
            'recommendations': self._get_recommendations()
        }
        
        # Save report
        report_file = self.repo_root / "reports" / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"📊 Cleanup report saved: {report_file}")
        
        # Print summary
        self._print_summary(report)
        
        return report
        
    def _get_recommendations(self) -> List[str]:
        """Get cleanup recommendations"""
        recommendations = []
        
        if self.stats['evolution_files_found'] > 1000:
            recommendations.append("Consider implementing automatic evolution state cleanup")
            
        if (self.repo_root / "node_modules").exists():
            recommendations.append("Consider using .dockerignore to exclude node_modules from builds")
            
        if len(list(self.repo_root.glob("*.log"))) > 10:
            recommendations.append("Implement log rotation to prevent log file accumulation")
            
        recommendations.extend([
            "Set up .gitignore patterns for evolution state files",
            "Implement periodic cleanup automation",
            "Consider using Git LFS for large data files"
        ])
        
        return recommendations
        
    def _print_summary(self, report: Dict[str, Any]):
        """Print cleanup summary"""
        stats = report['statistics']
        
        print("\n" + "="*60)
        print("🧹 REPOSITORY CLEANUP SUMMARY")
        print("="*60)
        print(f"📁 Directories created: {stats['directories_created']}")
        print(f"📦 Evolution files archived: {stats['evolution_files_archived']}")
        print(f"🔄 Files moved: {stats['files_moved']}")
        print(f"💾 Space saved: {report['space_saved_mb']} MB")
        print(f"📊 Archives created: {report['archives_created']}")
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        print("="*60 + "\n")

def main():
    """Main cleanup function"""
    cleanup = RepositoryCleanup()
    
    try:
        report = cleanup.run_cleanup()
        return report
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise

if __name__ == "__main__":
    main()