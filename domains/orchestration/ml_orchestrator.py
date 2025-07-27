#!/usr/bin/env python3

import asyncio
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

try:
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    logging.warning("ML dependencies not installed. Using fallback prediction.")

from orchestration.orchestrator import Orchestrator, Campaign, CampaignStatus, CampaignPriority
from orchestration.scheduler import CampaignScheduler
from knowledge_fabric.atom import KnowledgeAtom, AtomType


@dataclass
class TargetFeatures:
    hostname: str
    domain_age_days: int = 0
    tech_stack_complexity: float = 0.0
    security_headers_score: float = 0.0
    previous_findings_count: int = 0
    bounty_program_exists: bool = False
    last_scan_days_ago: int = 999
    subdomain_count: int = 0
    port_count: int = 0
    ssl_score: float = 0.0
    waf_detected: bool = False
    cdn_usage: bool = False
    industry_category: str = "unknown"
    domain_tld: str = "com"
    response_time_ms: float = 1000.0


@dataclass
class CampaignPerformanceMetrics:
    campaign_id: str
    target_hostname: str
    start_time: datetime
    end_time: Optional[datetime] = None
    findings_count: int = 0
    high_severity_findings: int = 0
    success_rate: float = 0.0
    resource_efficiency: float = 0.0
    monetization_value: float = 0.0
    agent_hours_used: float = 0.0


class MLTargetPredictor:
    """XGBoost-based target value prediction"""
    
    def __init__(self, model_path: str = "./models/target_predictor.pkl"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model: Optional[xgb.XGBRegressor] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        
        self.feature_columns = [
            'domain_age_days', 'tech_stack_complexity', 'security_headers_score',
            'previous_findings_count', 'bounty_program_exists', 'last_scan_days_ago',
            'subdomain_count', 'port_count', 'ssl_score', 'waf_detected',
            'cdn_usage', 'industry_category', 'domain_tld', 'response_time_ms'
        ]
        
        self.categorical_features = ['industry_category', 'domain_tld']
        
        self.logger = logging.getLogger(__name__)
        
        if HAS_ML_DEPS:
            self._initialize_model()
        else:
            self.logger.warning("ML dependencies not available. Using heuristic predictions.")

    def _initialize_model(self):
        """Initialize XGBoost model with CPU-optimized parameters"""
        if not HAS_ML_DEPS:
            return
            
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=4,  # CPU-friendly depth
            n_estimators=50,  # Lightweight model
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=2,  # Use 2 CPU cores max
            tree_method='hist'  # CPU-optimized method
        )
        
        # Try to load existing model
        if self.model_path.exists():
            try:
                self._load_model()
                self.logger.info("Loaded existing ML target prediction model")
            except Exception as e:
                self.logger.warning(f"Failed to load model: {e}. Will train new model.")

    async def predict_target_value(self, features: TargetFeatures) -> float:
        """Predict expected value (0-1) for a target"""
        if not HAS_ML_DEPS or self.model is None:
            return self._heuristic_prediction(features)
        
        try:
            # Convert features to DataFrame
            feature_dict = {
                'domain_age_days': features.domain_age_days,
                'tech_stack_complexity': features.tech_stack_complexity,
                'security_headers_score': features.security_headers_score,
                'previous_findings_count': features.previous_findings_count,
                'bounty_program_exists': int(features.bounty_program_exists),
                'last_scan_days_ago': features.last_scan_days_ago,
                'subdomain_count': features.subdomain_count,
                'port_count': features.port_count,
                'ssl_score': features.ssl_score,
                'waf_detected': int(features.waf_detected),
                'cdn_usage': int(features.cdn_usage),
                'industry_category': features.industry_category,
                'domain_tld': features.domain_tld,
                'response_time_ms': features.response_time_ms
            }
            
            df = pd.DataFrame([feature_dict])
            
            # Encode categorical features
            for feature in self.categorical_features:
                if feature in self.label_encoders:
                    try:
                        df[feature] = self.label_encoders[feature].transform(df[feature])
                    except ValueError:
                        # Unknown category, use most common class
                        df[feature] = 0
                else:
                    # First time seeing this feature, create encoder
                    self.label_encoders[feature] = LabelEncoder()
                    df[feature] = 0  # Default for unknown
            
            # Scale numerical features
            numerical_features = [f for f in self.feature_columns if f not in self.categorical_features]
            if hasattr(self.scaler, 'mean_'):  # Scaler is fitted
                df[numerical_features] = self.scaler.transform(df[numerical_features])
            
            # Predict
            prediction = self.model.predict(df[self.feature_columns])[0]
            
            # Ensure prediction is in [0, 1] range
            prediction = max(0.0, min(1.0, prediction))
            
            self.logger.debug(f"ML prediction for {features.hostname}: {prediction:.3f}")
            return prediction
            
        except Exception as e:
            self.logger.error(f"ML prediction failed for {features.hostname}: {e}")
            return self._heuristic_prediction(features)

    def _heuristic_prediction(self, features: TargetFeatures) -> float:
        """Fallback heuristic-based prediction"""
        score = 0.5  # Base score
        
        # Domain age factor (older domains often have more vulnerabilities)
        if features.domain_age_days > 365 * 5:  # > 5 years
            score += 0.1
        elif features.domain_age_days < 365:  # < 1 year
            score -= 0.1
        
        # Tech stack complexity
        score += min(0.2, features.tech_stack_complexity * 0.2)
        
        # Security headers (lower score = more vulnerable)
        score += (1.0 - features.security_headers_score) * 0.15
        
        # Previous findings indicate potential
        score += min(0.25, features.previous_findings_count * 0.05)
        
        # Bounty program availability
        if features.bounty_program_exists:
            score += 0.15
        
        # Recent scans (fresher targets might have new vulnerabilities)
        if features.last_scan_days_ago > 90:
            score += 0.1
        
        # WAF detection (might indicate more valuable target)
        if features.waf_detected:
            score += 0.05
        
        # Subdomain count (larger attack surface)
        score += min(0.1, features.subdomain_count * 0.01)
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, score))

    async def train_model(self, training_data: List[Tuple[TargetFeatures, float]]):
        """Train the ML model with historical data"""
        if not HAS_ML_DEPS or not training_data:
            self.logger.warning("Cannot train model: missing dependencies or data")
            return
        
        try:
            # Prepare training data
            feature_dicts = []
            targets = []
            
            for features, target_value in training_data:
                feature_dict = {
                    'domain_age_days': features.domain_age_days,
                    'tech_stack_complexity': features.tech_stack_complexity,
                    'security_headers_score': features.security_headers_score,
                    'previous_findings_count': features.previous_findings_count,
                    'bounty_program_exists': int(features.bounty_program_exists),
                    'last_scan_days_ago': features.last_scan_days_ago,
                    'subdomain_count': features.subdomain_count,
                    'port_count': features.port_count,
                    'ssl_score': features.ssl_score,
                    'waf_detected': int(features.waf_detected),
                    'cdn_usage': int(features.cdn_usage),
                    'industry_category': features.industry_category,
                    'domain_tld': features.domain_tld,
                    'response_time_ms': features.response_time_ms
                }
                feature_dicts.append(feature_dict)
                targets.append(target_value)
            
            df = pd.DataFrame(feature_dicts)
            
            # Encode categorical features
            for feature in self.categorical_features:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(df[feature])
            
            # Scale numerical features
            numerical_features = [f for f in self.feature_columns if f not in self.categorical_features]
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
            
            # Split data
            X = df[self.feature_columns]
            y = np.array(targets)
            
            if len(X) < 10:
                self.logger.warning(f"Insufficient training data: {len(X)} samples")
                return
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            self.logger.info(f"Model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")

    def _save_model(self):
        """Save model and encoders"""
        try:
            model_data = {
                'model': self.model,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'categorical_features': self.categorical_features
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    def _load_model(self):
        """Load model and encoders"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data.get('feature_columns', self.feature_columns)
            self.categorical_features = model_data.get('categorical_features', self.categorical_features)
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise


class AdaptiveCampaignManager:
    """Manages campaign adaptation based on real-time performance"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.performance_history: List[CampaignPerformanceMetrics] = []
        self.adaptation_threshold = 0.3  # Adapt if success rate drops below 30%
        self.logger = logging.getLogger(__name__)

    async def monitor_campaign_performance(self, campaign_id: str):
        """Monitor and adapt campaign in real-time"""
        while True:
            try:
                campaign = self.orchestrator.campaigns.get(campaign_id)
                if not campaign or campaign.status != CampaignStatus.RUNNING:
                    break
                
                # Calculate current performance metrics
                metrics = await self._calculate_performance_metrics(campaign)
                self.performance_history.append(metrics)
                
                # Check if adaptation is needed
                if await self._needs_adaptation(campaign, metrics):
                    await self._adapt_campaign(campaign, metrics)
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error monitoring campaign {campaign_id}: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute

    async def _calculate_performance_metrics(self, campaign: Campaign) -> CampaignPerformanceMetrics:
        """Calculate current campaign performance"""
        findings_count = len(campaign.findings)
        high_severity_count = len([f for f in campaign.findings if f.get('severity', '').lower() in ['high', 'critical']])
        
        # Calculate success rate based on findings vs time invested
        time_elapsed = (datetime.utcnow() - campaign.started_at).total_seconds() / 3600 if campaign.started_at else 1
        success_rate = findings_count / max(time_elapsed, 1)  # findings per hour
        
        # Calculate resource efficiency
        active_agents = len([agents for agents in campaign.agent_assignments.values() for agent in agents])
        resource_efficiency = findings_count / max(active_agents, 1)
        
        # Estimate monetization value
        monetization_value = high_severity_count * 500 + (findings_count - high_severity_count) * 100  # Rough estimate
        
        return CampaignPerformanceMetrics(
            campaign_id=campaign.id,
            target_hostname=campaign.targets[0].hostname if campaign.targets else "unknown",
            start_time=campaign.started_at or datetime.utcnow(),
            findings_count=findings_count,
            high_severity_findings=high_severity_count,
            success_rate=success_rate,
            resource_efficiency=resource_efficiency,
            monetization_value=monetization_value,
            agent_hours_used=time_elapsed * active_agents
        )

    async def _needs_adaptation(self, campaign: Campaign, metrics: CampaignPerformanceMetrics) -> bool:
        """Determine if campaign needs adaptation"""
        # Check if success rate is below threshold
        if metrics.success_rate < self.adaptation_threshold:
            return True
        
        # Check if no findings in the last hour
        recent_findings = [f for f in campaign.findings 
                          if datetime.fromisoformat(f.get('timestamp', '2000-01-01')) > 
                          datetime.utcnow() - timedelta(hours=1)]
        
        if len(recent_findings) == 0 and metrics.agent_hours_used > 2:
            return True
        
        return False

    async def _adapt_campaign(self, campaign: Campaign, metrics: CampaignPerformanceMetrics):
        """Adapt campaign strategy based on performance"""
        self.logger.info(f"Adapting campaign {campaign.id} - Success rate: {metrics.success_rate:.3f}")
        
        # Strategy 1: Reduce agent allocation for poor performers
        if metrics.success_rate < 0.1:
            await self._reduce_agent_allocation(campaign, reduction_factor=0.5)
            await self._switch_to_conservative_tactics(campaign)
        
        # Strategy 2: Try different agent types
        elif metrics.success_rate < 0.2:
            await self._rotate_agent_types(campaign)
        
        # Strategy 3: Adjust scanning intensity
        elif metrics.success_rate < 0.3:
            await self._adjust_scanning_intensity(campaign, intensity_factor=0.7)

    async def _reduce_agent_allocation(self, campaign: Campaign, reduction_factor: float):
        """Reduce number of agents assigned to campaign"""
        for agent_type, agent_list in campaign.agent_assignments.items():
            if len(agent_list) > 1:  # Keep at least one agent
                reduce_count = max(1, int(len(agent_list) * (1 - reduction_factor)))
                agents_to_remove = agent_list[:reduce_count]
                
                for agent_id in agents_to_remove:
                    agent_list.remove(agent_id)
                    self.logger.debug(f"Removed agent {agent_id} from campaign {campaign.id}")

    async def _switch_to_conservative_tactics(self, campaign: Campaign):
        """Switch to more conservative, targeted tactics"""
        # Add metadata to indicate conservative mode
        campaign.metadata['tactic_mode'] = 'conservative'
        campaign.metadata['scan_intensity'] = 'low'
        campaign.metadata['adapted_at'] = datetime.utcnow().isoformat()
        
        self.logger.info(f"Switched campaign {campaign.id} to conservative tactics")

    async def _rotate_agent_types(self, campaign: Campaign):
        """Try different types of agents"""
        available_types = ['recon', 'web_crawler', 'vulnerability_scanner', 'nuclei_scanner']
        current_types = set(campaign.agent_assignments.keys())
        
        # Try agent types not currently in use
        unused_types = [t for t in available_types if t not in current_types]
        
        if unused_types:
            new_agent_type = unused_types[0]  # Try first unused type
            new_agent_id = f"{new_agent_type}_{campaign.id[:8]}"
            campaign.agent_assignments[new_agent_type] = [new_agent_id]
            
            self.logger.info(f"Added {new_agent_type} agent to campaign {campaign.id}")

    async def _adjust_scanning_intensity(self, campaign: Campaign, intensity_factor: float):
        """Adjust scanning intensity/frequency"""
        campaign.metadata['scan_intensity_factor'] = intensity_factor
        campaign.metadata['intensity_adjusted_at'] = datetime.utcnow().isoformat()
        
        self.logger.info(f"Adjusted scanning intensity to {intensity_factor} for campaign {campaign.id}")


class IntelligentOrchestrator(Orchestrator):
    """Enhanced orchestrator with ML-powered decision making"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        super().__init__(redis_url)
        
        self.ml_predictor = MLTargetPredictor()
        self.adaptive_manager = AdaptiveCampaignManager(self)
        self.performance_metrics: List[CampaignPerformanceMetrics] = []
        
        # Enhanced configuration
        self.ml_enabled = True
        self.adaptive_campaigns = True
        self.prediction_threshold = 0.6  # Only target if prediction > 0.6

    async def create_intelligent_campaign(self, 
                                        name: str, 
                                        targets: List[Dict], 
                                        priority: CampaignPriority = CampaignPriority.MEDIUM,
                                        metadata: Optional[Dict] = None) -> str:
        """Create campaign with ML-powered target analysis"""
        
        # Analyze and score all targets
        scored_targets = []
        
        for target_data in targets:
            # Extract features for this target
            features = await self._extract_target_features(target_data)
            
            # Predict target value
            prediction_score = await self.ml_predictor.predict_target_value(features)
            
            # Only include targets above threshold
            if prediction_score >= self.prediction_threshold:
                target_data['ml_score'] = prediction_score
                target_data['prediction_confidence'] = min(1.0, prediction_score * 1.2)
                scored_targets.append(target_data)
                
                self.logger.info(f"Target {target_data.get('hostname', 'unknown')} scored {prediction_score:.3f}")
            else:
                self.logger.info(f"Target {target_data.get('hostname', 'unknown')} filtered out (score: {prediction_score:.3f})")
        
        if not scored_targets:
            self.logger.warning("No targets met ML prediction threshold")
            # Fall back to original targets with warning
            scored_targets = targets
        
        # Sort targets by ML score (highest first)
        scored_targets.sort(key=lambda x: x.get('ml_score', 0.5), reverse=True)
        
        # Create campaign with enhanced metadata
        enhanced_metadata = metadata or {}
        enhanced_metadata.update({
            'ml_enabled': True,
            'prediction_threshold': self.prediction_threshold,
            'targets_analyzed': len(targets),
            'targets_selected': len(scored_targets),
            'ml_model_version': '1.0'
        })
        
        # Create campaign using parent method
        campaign_id = await self.create_campaign(name, scored_targets, priority, enhanced_metadata)
        
        # Start adaptive monitoring for this campaign
        if self.adaptive_campaigns:
            asyncio.create_task(self.adaptive_manager.monitor_campaign_performance(campaign_id))
        
        return campaign_id

    async def _extract_target_features(self, target_data: Dict) -> TargetFeatures:
        """Extract ML features from target data"""
        hostname = target_data.get('hostname', '')
        
        # Basic features from target data
        features = TargetFeatures(
            hostname=hostname,
            subdomain_count=len(target_data.get('subdomains', [])),
            port_count=len(target_data.get('ports', [])),
            domain_tld=hostname.split('.')[-1] if '.' in hostname else 'unknown'
        )
        
        # Try to enrich with additional intelligence
        try:
            # Get domain age (simplified - in production would use WHOIS)
            domain_parts = hostname.split('.')
            if len(domain_parts) >= 2:
                # Heuristic: shorter domains are often older
                features.domain_age_days = max(365, (10 - len(hostname)) * 365)
            
            # Technology stack complexity (from previous scans or knowledge base)
            tech_info = await self._get_technology_info(hostname)
            features.tech_stack_complexity = len(tech_info.get('technologies', [])) / 10.0
            
            # Security headers score (from previous scans)
            security_info = await self._get_security_info(hostname)
            features.security_headers_score = security_info.get('security_score', 0.5)
            
            # Previous findings count
            features.previous_findings_count = len(await self._get_previous_findings(hostname))
            
            # Check if bounty program exists
            features.bounty_program_exists = await self._check_bounty_program(hostname)
            
            # Industry categorization (simplified)
            features.industry_category = self._categorize_domain(hostname)
            
        except Exception as e:
            self.logger.debug(f"Feature extraction partially failed for {hostname}: {e}")
        
        return features

    async def _get_technology_info(self, hostname: str) -> Dict:
        """Get technology stack information for hostname"""
        # In production, this would query knowledge base or external services
        # For now, return mock data based on domain characteristics
        
        tech_indicators = {
            'wp-': ['WordPress'],
            'shop': ['E-commerce'],
            'api': ['REST API'],
            'admin': ['Admin Panel'],
            'mail': ['Email Service'],
            'cdn': ['CDN']
        }
        
        technologies = []
        for indicator, techs in tech_indicators.items():
            if indicator in hostname.lower():
                technologies.extend(techs)
        
        return {'technologies': technologies}

    async def _get_security_info(self, hostname: str) -> Dict:
        """Get security information for hostname"""
        # Mock security score based on domain characteristics
        # In production, would query security headers, SSL config, etc.
        
        score = 0.5  # Base score
        
        if 'https' in hostname:
            score += 0.2
        if len(hostname) > 20:  # Longer domains often have better security
            score += 0.1
        if any(word in hostname for word in ['secure', 'safe', 'protected']):
            score += 0.2
        
        return {'security_score': min(1.0, score)}

    async def _get_previous_findings(self, hostname: str) -> List[Dict]:
        """Get previous security findings for hostname"""
        # Query knowledge base for previous findings
        try:
            atoms = await self.knowledge_fabric.search_atoms(
                query=hostname,
                atom_type=AtomType.VULNERABILITY,
                max_results=50
            )
            
            findings = []
            for atom in atoms:
                if hostname in str(atom.content):
                    findings.append({
                        'id': atom.id,
                        'title': atom.title,
                        'severity': atom.content.get('severity', 'unknown'),
                        'found_at': atom.created_at
                    })
            
            return findings
            
        except Exception as e:
            self.logger.debug(f"Failed to get previous findings for {hostname}: {e}")
            return []

    async def _check_bounty_program(self, hostname: str) -> bool:
        """Check if hostname has active bounty programs"""
        # In production, would check HackerOne, Bugcrowd, etc.
        # For now, simple heuristics
        
        bounty_indicators = [
            'facebook', 'google', 'microsoft', 'apple', 'twitter',
            'github', 'dropbox', 'uber', 'airbnb', 'netflix'
        ]
        
        return any(indicator in hostname.lower() for indicator in bounty_indicators)

    def _categorize_domain(self, hostname: str) -> str:
        """Categorize domain by industry"""
        categories = {
            'tech': ['api', 'dev', 'github', 'gitlab', 'tech'],
            'finance': ['bank', 'pay', 'finance', 'money', 'credit'],
            'ecommerce': ['shop', 'store', 'buy', 'cart', 'commerce'],
            'social': ['social', 'chat', 'forum', 'community'],
            'media': ['news', 'media', 'blog', 'content'],
            'education': ['edu', 'school', 'university', 'learn'],
            'healthcare': ['health', 'medical', 'hospital', 'clinic']
        }
        
        hostname_lower = hostname.lower()
        
        for category, keywords in categories.items():
            if any(keyword in hostname_lower for keyword in keywords):
                return category
        
        return 'unknown'

    async def retrain_prediction_model(self):
        """Retrain ML model with recent campaign results"""
        if not self.ml_enabled:
            return
        
        self.logger.info("Starting ML model retraining")
        
        # Collect training data from recent campaigns
        training_data = []
        
        for metrics in self.performance_metrics[-100:]:  # Last 100 campaigns
            # Calculate actual value based on campaign results
            actual_value = min(1.0, (metrics.findings_count * 0.1) + 
                              (metrics.high_severity_findings * 0.3) + 
                              (metrics.monetization_value / 1000))
            
            # Reconstruct features (in production, would store features)
            features = await self._reconstruct_target_features(metrics.target_hostname)
            
            training_data.append((features, actual_value))
        
        if len(training_data) >= 10:  # Need minimum data for training
            await self.ml_predictor.train_model(training_data)
            self.logger.info(f"Retrained ML model with {len(training_data)} samples")
        else:
            self.logger.warning(f"Insufficient data for retraining: {len(training_data)} samples")

    async def _reconstruct_target_features(self, hostname: str) -> TargetFeatures:
        """Reconstruct target features for training"""
        # This is a simplified reconstruction
        # In production, would store original features with campaign data
        return await self._extract_target_features({'hostname': hostname})

    async def get_ml_orchestrator_stats(self) -> Dict[str, Any]:
        """Get ML orchestrator statistics"""
        return {
            'ml_enabled': self.ml_enabled,
            'adaptive_campaigns': self.adaptive_campaigns,
            'prediction_threshold': self.prediction_threshold,
            'model_available': self.ml_predictor.model is not None,
            'total_campaigns': len(self.campaigns),
            'performance_metrics_count': len(self.performance_metrics),
            'has_ml_dependencies': HAS_ML_DEPS,
            'avg_prediction_accuracy': await self._calculate_prediction_accuracy(),
            'adaptation_events': await self._count_adaptation_events()
        }

    async def _calculate_prediction_accuracy(self) -> float:
        """Calculate ML prediction accuracy"""
        if len(self.performance_metrics) < 5:
            return 0.0
        
        # Compare predictions with actual results
        accuracies = []
        
        for metrics in self.performance_metrics[-20:]:  # Last 20 campaigns
            predicted_value = 0.7  # Mock - in production would store prediction
            actual_value = min(1.0, metrics.success_rate / 2.0)  # Normalize
            
            accuracy = 1.0 - abs(predicted_value - actual_value)
            accuracies.append(accuracy)
        
        return np.mean(accuracies) if accuracies else 0.0

    async def _count_adaptation_events(self) -> int:
        """Count how many campaigns required adaptation"""
        count = 0
        
        for campaign in self.campaigns.values():
            if campaign.metadata.get('tactic_mode') == 'conservative':
                count += 1
        
        return count


# Install ML dependencies helper
def check_and_install_ml_deps():
    """Check and optionally install ML dependencies"""
    try:
        import xgboost
        import sklearn
        return True
    except ImportError:
        print("ML dependencies not found. Install with:")
        print("pip install xgboost scikit-learn pandas numpy")
        return False


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if "--check-deps" in sys.argv:
        if check_and_install_ml_deps():
            print("✅ All ML dependencies are available")
        else:
            print("❌ ML dependencies missing")
        sys.exit(0)
    
    async def main():
        orchestrator = IntelligentOrchestrator()
        await orchestrator.start()
        
        # Test with sample targets
        test_targets = [
            {"hostname": "example.com", "ports": [80, 443]},
            {"hostname": "test-api.example.com", "ports": [443, 8080]},
            {"hostname": "admin.vulnerable-site.org", "ports": [80, 443, 8080, 3306]}
        ]
        
        try:
            campaign_id = await orchestrator.create_intelligent_campaign(
                name="ML-Powered Test Campaign",
                targets=test_targets,
                priority=CampaignPriority.HIGH
            )
            
            print(f"Created intelligent campaign: {campaign_id}")
            
            # Start campaign
            await orchestrator.start_campaign(campaign_id)
            
            # Show stats
            stats = await orchestrator.get_ml_orchestrator_stats()
            print(f"ML Orchestrator Stats: {stats}")
            
            # Keep running for demo
            print("Press Ctrl+C to stop")
            while True:
                await asyncio.sleep(10)
                
                # Show campaign status
                status = await orchestrator.get_campaign_status(campaign_id)
                if status:
                    print(f"Campaign Status: {status['status']} - Findings: {status['findings_count']}")
        
        except KeyboardInterrupt:
            print("\nStopping orchestrator...")
        finally:
            await orchestrator.shutdown()
    
    asyncio.run(main())