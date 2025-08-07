import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from xorb.shared.epyc_config import EPYCConfig
from xorb.shared.enums import AgentType
from xorb.shared.models import UnifiedTarget

# ML-powered Campaign Planner
class MLCampaignPlanner:
    def __init__(self):
        self.device = torch.device('cpu')  # EPYC CPU-only
        self.tokenizer = None
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize ML models for campaign planning."""
        try:
            # Use lightweight model for EPYC deployment
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Set EPYC optimization
            torch.set_num_threads(EPYCConfig.TORCH_NUM_THREADS)
            
            self.logger.info("ML Campaign Planner initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
    
    async def generate_campaign_plan(self, targets: List[UnifiedTarget], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent campaign plan using ML."""
        try:
            # Extract target features
            target_features = []
            for target in targets:
                features = {
                    'ports_count': len(target.ports),
                    'services_count': len(target.services),
                    'vuln_count': len(target.vulnerabilities),
                    'confidence': target.confidence
                }
                target_features.append(features)
            
            # Generate plan phases
            phases = []
            
            # Phase 1: Reconnaissance
            recon_phase = {
                'phase': 'reconnaissance',
                'duration_estimate': 300,  # 5 minutes
                'agent_types': [AgentType.RECONNAISSANCE.value],
                'priority': 'high'
            }
            phases.append(recon_phase)
            
            # Phase 2: Scanning (based on target complexity)
            scan_duration = min(600, len(targets) * 120)  # Max 10 min
            scan_phase = {
                'phase': 'scanning',
                'duration_estimate': scan_duration,
                'agent_types': [AgentType.SCANNING.value],
                'priority': 'medium'
            }
            phases.append(scan_phase)
            
            # Phase 3: Exploitation (if authorized)
            if requirements.get('allow_exploitation', False):
                exploit_phase = {
                    'phase': 'exploitation',
                    'duration_estimate': 900,  # 15 minutes
                    'agent_types': [AgentType.EXPLOITATION.value, AgentType.STEALTH.value],
                    'priority': 'high'
                }
                phases.append(exploit_phase)
            
            return {
                'phases': phases,
                'total_duration_estimate': sum(p['duration_estimate'] for p in phases),
                'required_agents': sum([p['agent_types'] for p in phases], []),
                'risk_level': self._calculate_risk_level(targets),
                'success_probability': self._estimate_success_probability(target_features)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating campaign plan: {e}")
            return {'error': str(e)}
    
    def _calculate_risk_level(self, targets: List[UnifiedTarget]) -> str:
        """Calculate campaign risk level."""
        total_vulns = sum(len(t.vulnerabilities) for t in targets)
        if total_vulns > 10:
            return 'high'
        elif total_vulns > 5:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_success_probability(self, target_features: List[Dict]) -> float:
        """Estimate campaign success probability."""
        if not target_features:
            return 0.5
        
        # Simple heuristic based on target complexity
        avg_complexity = np.mean([
            f['ports_count'] + f['services_count'] + f['vuln_count'] 
            for f in target_features
        ])
        
        # Higher complexity = higher success probability (more attack surface)
        return min(0.95, 0.3 + (avg_complexity * 0.05))