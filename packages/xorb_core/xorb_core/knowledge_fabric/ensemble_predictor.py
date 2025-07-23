#!/usr/bin/env python3

import logging
import pickle
import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    logging.warning("Advanced ML libraries not available. Install with: pip install xgboost lightgbm catboost scikit-learn")

from .atom import KnowledgeAtom, AtomType, ConfidenceLevel
from .ml_predictor import PredictionFeatures, ConfidencePredictor


@dataclass
class EnsemblePredictionResult:
    value: float
    confidence_interval: Tuple[float, float]
    model_predictions: Dict[str, float]
    feature_importance: Dict[str, float]
    prediction_quality: str  # 'high', 'medium', 'low'


class EnsembleTargetPredictor:
    """Advanced ensemble ML predictor combining multiple algorithms for superior accuracy."""
    
    def __init__(self, model_dir: str = "./models/ensemble"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.meta_learner = None
        self.feature_scaler = None
        self.is_trained = False
        
        self.logger = logging.getLogger(__name__)
        
        # Fallback to simple predictor if advanced ML not available
        self.fallback_predictor = ConfidencePredictor()
        
        if ADVANCED_ML_AVAILABLE:
            self._initialize_ensemble_models()
        else:
            self.logger.warning("Using fallback predictor due to missing ML dependencies")
    
    def _initialize_ensemble_models(self):
        """Initialize ensemble models optimized for CPU performance."""
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,  # Use all CPU cores
                tree_method='hist'  # Faster CPU training
            ),
            'lightgbm': lgb.LGBMRegressor(
                num_leaves=31,
                n_estimators=100,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                task_type='CPU',
                verbose=False
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Meta-learner for stacking ensemble
        self.meta_learner = LinearRegression()
    
    async def initialize(self):
        """Initialize the ensemble predictor."""
        await self.fallback_predictor.initialize()
        
        if ADVANCED_ML_AVAILABLE:
            await self._load_ensemble_models()
        
        self.logger.info("Ensemble predictor initialized")
    
    async def predict_target_value(self, atom: KnowledgeAtom, 
                                 context: Dict[str, Any] = None) -> EnsemblePredictionResult:
        """Predict target value using ensemble of models."""
        features = await self._extract_advanced_features(atom, context)
        
        if not ADVANCED_ML_AVAILABLE or not self.is_trained:
            # Fallback to simple predictor
            fallback_pred = await self.fallback_predictor.predict_confidence(atom)
            return EnsemblePredictionResult(
                value=fallback_pred,
                confidence_interval=(max(0, fallback_pred - 0.1), min(1, fallback_pred + 0.1)),
                model_predictions={'fallback': fallback_pred},
                feature_importance={},
                prediction_quality='medium'
            )
        
        # Get predictions from all models
        model_predictions = {}
        feature_vector = self._features_to_vector(features)
        
        try:
            for name, model in self.models.items():
                if hasattr(model, 'predict'):
                    pred = model.predict([feature_vector])[0]
                    model_predictions[name] = float(pred)
        except Exception as e:
            self.logger.warning(f"Model prediction failed: {e}")
            # Fallback
            fallback_pred = await self.fallback_predictor.predict_confidence(atom)
            return EnsemblePredictionResult(
                value=fallback_pred,
                confidence_interval=(max(0, fallback_pred - 0.15), min(1, fallback_pred + 0.15)),
                model_predictions={'fallback': fallback_pred},
                feature_importance={},
                prediction_quality='low'
            )
        
        # Ensemble prediction using meta-learner
        if self.meta_learner and len(model_predictions) > 1:
            ensemble_input = list(model_predictions.values())
            ensemble_pred = self.meta_learner.predict([ensemble_input])[0]
        else:
            # Simple averaging
            ensemble_pred = np.mean(list(model_predictions.values()))
        
        # Calculate confidence interval based on model variance
        pred_std = np.std(list(model_predictions.values()))
        confidence_interval = (
            max(0.0, ensemble_pred - 1.96 * pred_std),
            min(1.0, ensemble_pred + 1.96 * pred_std)
        )
        
        # Feature importance (from XGBoost if available)
        feature_importance = {}
        if 'xgboost' in model_predictions and hasattr(self.models['xgboost'], 'feature_importances_'):
            feature_names = self._get_feature_names()
            importance_scores = self.models['xgboost'].feature_importances_
            feature_importance = dict(zip(feature_names, importance_scores))
        
        # Determine prediction quality
        quality = 'high' if pred_std < 0.1 else 'medium' if pred_std < 0.2 else 'low'
        
        return EnsemblePredictionResult(
            value=float(ensemble_pred),
            confidence_interval=confidence_interval,
            model_predictions=model_predictions,
            feature_importance=feature_importance,
            prediction_quality=quality
        )
    
    async def predict_success_probability(self, atom: KnowledgeAtom, 
                                        target_context: Dict[str, Any]) -> float:
        """Predict success probability for specific target context."""
        result = await self.predict_target_value(atom, target_context)
        
        # Adjust for target-specific factors
        base_probability = result.value
        
        # Context-specific adjustments
        if target_context:
            target_type = target_context.get('target_type', 'unknown')
            difficulty = target_context.get('difficulty_score', 0.5)
            security_posture = target_context.get('security_score', 0.5)
            
            # Adjust based on target characteristics
            context_multiplier = 1.0
            
            if atom.atom_type == AtomType.VULNERABILITY:
                if target_type == 'web_application':
                    context_multiplier = 1.2
                elif target_type == 'api':
                    context_multiplier = 1.1
                elif target_type == 'network':
                    context_multiplier = 0.9
            
            # Adjust for security posture
            security_adjustment = 1.0 - (security_posture * 0.3)
            context_multiplier *= security_adjustment
            
            # Adjust for difficulty
            difficulty_adjustment = 1.0 - (difficulty * 0.2)
            context_multiplier *= difficulty_adjustment
            
            base_probability *= context_multiplier
        
        return max(0.0, min(1.0, base_probability))
    
    async def train_ensemble(self, training_data: List[Tuple[KnowledgeAtom, float]], 
                           validation_split: float = 0.2):
        """Train the ensemble models with historical data."""
        if not ADVANCED_ML_AVAILABLE:
            self.logger.warning("Cannot train ensemble without ML libraries")
            return
        
        if len(training_data) < 50:
            self.logger.warning(f"Insufficient training data: {len(training_data)} samples")
            return
        
        self.logger.info(f"Training ensemble with {len(training_data)} samples")
        
        # Prepare training data
        X, y = await self._prepare_training_data(training_data)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train individual models
        model_scores = {}
        for name, model in self.models.items():
            try:
                self.logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Validate
                val_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, val_pred)
                model_scores[name] = mae
                
                self.logger.info(f"{name} validation MAE: {mae:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {name}: {e}")
                del self.models[name]
        
        # Train meta-learner if we have multiple models
        if len(self.models) > 1:
            self.logger.info("Training meta-learner...")
            
            # Get predictions from all models for meta-learning
            meta_X = []
            for i in range(len(X_train)):
                model_preds = []
                for model in self.models.values():
                    pred = model.predict([X_train[i]])[0]
                    model_preds.append(pred)
                meta_X.append(model_preds)
            
            self.meta_learner.fit(meta_X, y_train)
            
            # Validate meta-learner
            meta_val_X = []
            for i in range(len(X_val)):
                model_preds = []
                for model in self.models.values():
                    pred = model.predict([X_val[i]])[0]
                    model_preds.append(pred)
                meta_val_X.append(model_preds)
            
            meta_pred = self.meta_learner.predict(meta_val_X)
            meta_mae = mean_absolute_error(y_val, meta_pred)
            
            self.logger.info(f"Meta-learner validation MAE: {meta_mae:.4f}")
        
        self.is_trained = True
        await self._save_ensemble_models()
        
        self.logger.info("Ensemble training completed")
    
    async def _extract_advanced_features(self, atom: KnowledgeAtom, 
                                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract comprehensive features for ML prediction."""
        # Get base features from simple predictor
        base_features = await self.fallback_predictor._extract_features(atom)
        
        # Add advanced features
        advanced_features = {
            # Temporal features
            'age_hours': (datetime.utcnow() - atom.created_at).total_seconds() / 3600,
            'days_since_update': (datetime.utcnow() - atom.updated_at).days if atom.updated_at else 0,
            'update_frequency': atom.usage_count / max(1, (datetime.utcnow() - atom.created_at).days),
            
            # Content features
            'content_length': len(str(atom.content)),
            'content_complexity': len(str(atom.content).split()),
            'has_code_snippets': 1 if any(marker in str(atom.content).lower() 
                                        for marker in ['```', 'code', 'script', 'function']) else 0,
            
            # Relationship features
            'relationship_diversity': len(set(r.relationship_type for r in atom.related_atoms)) if atom.related_atoms else 0,
            'avg_relationship_strength': np.mean([r.strength for r in atom.related_atoms]) if atom.related_atoms else 0,
            
            # Source features
            'source_count': len(atom.sources),
            'avg_source_reliability': np.mean([s.reliability_score for s in atom.sources]) if atom.sources else 0.5,
            'has_official_source': 1 if any('official' in s.source_type.lower() for s in atom.sources) else 0,
            
            # Validation features
            'validation_count': len(atom.validation_results),
            'validation_success_rate': len([v for v in atom.validation_results if v.is_valid]) / max(1, len(atom.validation_results)),
            
            # Usage patterns
            'usage_trend': self._calculate_usage_trend(atom),
            'success_consistency': self._calculate_success_consistency(atom),
        }
        
        # Add context features if available
        if context:
            advanced_features.update({
                'target_difficulty': context.get('difficulty_score', 0.5),
                'target_security_score': context.get('security_score', 0.5),
                'target_type_web': 1 if context.get('target_type') == 'web_application' else 0,
                'target_type_api': 1 if context.get('target_type') == 'api' else 0,
                'target_type_network': 1 if context.get('target_type') == 'network' else 0,
                'campaign_automated': 1 if context.get('campaign_type') == 'automated' else 0,
                'has_bounty_program': 1 if context.get('has_bounty_program', False) else 0,
            })
        
        # Combine with base features
        all_features = {
            **advanced_features,
            'base_age_days': base_features.age_days,
            'base_usage_frequency': base_features.usage_frequency,
            'base_success_rate': base_features.success_rate,
            'base_source_reliability': base_features.source_reliability,
            'base_validation_score': base_features.validation_score,
            'base_relationship_count': base_features.relationship_count,
            'base_content_complexity': base_features.content_complexity,
        }
        
        # Add atom type one-hot encoding
        for i, atom_type in enumerate(AtomType):
            all_features[f'atom_type_{atom_type.value}'] = 1 if atom.atom_type == atom_type else 0
        
        return all_features
    
    def _features_to_vector(self, features: Dict[str, Any]) -> List[float]:
        """Convert feature dictionary to vector for model input."""
        feature_names = self._get_feature_names()
        return [float(features.get(name, 0)) for name in feature_names]
    
    def _get_feature_names(self) -> List[str]:
        """Get standardized feature names."""
        base_features = [
            'age_hours', 'days_since_update', 'update_frequency',
            'content_length', 'content_complexity', 'has_code_snippets',
            'relationship_diversity', 'avg_relationship_strength',
            'source_count', 'avg_source_reliability', 'has_official_source',
            'validation_count', 'validation_success_rate',
            'usage_trend', 'success_consistency',
            'target_difficulty', 'target_security_score',
            'target_type_web', 'target_type_api', 'target_type_network',
            'campaign_automated', 'has_bounty_program',
            'base_age_days', 'base_usage_frequency', 'base_success_rate',
            'base_source_reliability', 'base_validation_score',
            'base_relationship_count', 'base_content_complexity'
        ]
        
        # Add atom type features
        atom_type_features = [f'atom_type_{atom_type.value}' for atom_type in AtomType]
        
        return base_features + atom_type_features
    
    def _calculate_usage_trend(self, atom: KnowledgeAtom) -> float:
        """Calculate usage trend over time."""
        # Simplified trend calculation - in practice would use time series data
        if atom.usage_count == 0:
            return 0.0
        
        age_days = (datetime.utcnow() - atom.created_at).days
        if age_days == 0:
            return 1.0
        
        recent_usage_rate = atom.usage_count / age_days
        return min(1.0, recent_usage_rate)
    
    def _calculate_success_consistency(self, atom: KnowledgeAtom) -> float:
        """Calculate consistency of success over time."""
        # Simplified - in practice would track success over time windows
        if atom.usage_count < 3:
            return 0.5  # Unknown consistency
        
        # Use validation results as proxy for consistency
        if atom.validation_results:
            success_rate = len([v for v in atom.validation_results if v.is_valid]) / len(atom.validation_results)
            return success_rate
        
        return atom.success_rate
    
    async def _prepare_training_data(self, training_data: List[Tuple[KnowledgeAtom, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models."""
        X, y = [], []
        
        for atom, target_value in training_data:
            features = await self._extract_advanced_features(atom)
            feature_vector = self._features_to_vector(features)
            
            X.append(feature_vector)
            y.append(target_value)
        
        return np.array(X), np.array(y)
    
    async def _save_ensemble_models(self):
        """Save trained ensemble models."""
        try:
            model_data = {
                'models': {},
                'meta_learner': self.meta_learner,
                'is_trained': self.is_trained,
                'feature_names': self._get_feature_names()
            }
            
            # Save individual models
            for name, model in self.models.items():
                model_path = self.model_dir / f"{name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                model_data['models'][name] = str(model_path)
            
            # Save meta data
            meta_path = self.model_dir / "ensemble_meta.pkl"
            with open(meta_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Ensemble models saved to {self.model_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save ensemble models: {e}")
    
    async def _load_ensemble_models(self):
        """Load trained ensemble models."""
        try:
            meta_path = self.model_dir / "ensemble_meta.pkl"
            if not meta_path.exists():
                self.logger.info("No saved ensemble models found")
                return
            
            with open(meta_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load individual models
            for name, model_path in model_data['models'].items():
                if Path(model_path).exists():
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            self.meta_learner = model_data.get('meta_learner')
            self.is_trained = model_data.get('is_trained', False)
            
            self.logger.info(f"Loaded {len(self.models)} ensemble models")
            
        except Exception as e:
            self.logger.error(f"Failed to load ensemble models: {e}")


class GraphBasedTargetAnalyzer:
    """Graph-based analysis for target relationships and attack paths."""
    
    def __init__(self):
        try:
            import networkx as nx
            self.nx = nx
            self.graph = nx.DiGraph()
            self.GRAPH_AVAILABLE = True
        except ImportError:
            self.GRAPH_AVAILABLE = False
            logging.warning("NetworkX not available. Install with: pip install networkx")
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze_target_relationships(self, atoms: List[KnowledgeAtom]) -> Dict[str, Any]:
        """Analyze relationships between targets using graph algorithms."""
        if not self.GRAPH_AVAILABLE:
            return {'error': 'Graph analysis not available'}
        
        # Build graph from atoms
        self._build_knowledge_graph(atoms)
        
        analysis = {
            'centrality_analysis': self._calculate_centrality(),
            'clustering_analysis': self._find_clusters(),
            'path_analysis': self._analyze_attack_paths(),
            'recommendation': self._generate_recommendations()
        }
        
        return analysis
    
    def _build_knowledge_graph(self, atoms: List[KnowledgeAtom]):
        """Build NetworkX graph from knowledge atoms."""
        self.graph.clear()
        
        # Add nodes
        for atom in atoms:
            self.graph.add_node(
                atom.id,
                atom_type=atom.atom_type.value,
                confidence=atom.confidence,
                success_rate=atom.success_rate,
                usage_count=atom.usage_count
            )
        
        # Add edges
        for atom in atoms:
            for related in atom.related_atoms:
                if related.target_atom_id in [a.id for a in atoms]:
                    self.graph.add_edge(
                        atom.id,
                        related.target_atom_id,
                        weight=related.strength,
                        relationship_type=related.relationship_type
                    )
    
    def _calculate_centrality(self) -> Dict[str, Any]:
        """Calculate node centrality metrics."""
        if len(self.graph.nodes) == 0:
            return {}
        
        return {
            'betweenness': dict(self.nx.betweenness_centrality(self.graph)),
            'closeness': dict(self.nx.closeness_centrality(self.graph)),
            'pagerank': dict(self.nx.pagerank(self.graph))
        }
    
    def _find_clusters(self) -> Dict[str, Any]:
        """Find clusters in the knowledge graph."""
        if len(self.graph.nodes) == 0:
            return {}
        
        # Convert to undirected for clustering
        undirected = self.graph.to_undirected()
        
        try:
            communities = list(self.nx.community.greedy_modularity_communities(undirected))
            return {
                'community_count': len(communities),
                'communities': [list(community) for community in communities],
                'modularity': self.nx.community.modularity(undirected, communities)
            }
        except:
            return {'error': 'Clustering failed'}
    
    def _analyze_attack_paths(self) -> Dict[str, Any]:
        """Analyze potential attack paths."""
        if len(self.graph.nodes) == 0:
            return {}
        
        # Find paths between high-value nodes
        vulnerability_nodes = [n for n, d in self.graph.nodes(data=True) 
                             if d.get('atom_type') == 'vulnerability']
        exploit_nodes = [n for n, d in self.graph.nodes(data=True) 
                        if d.get('atom_type') == 'exploit']
        
        paths = []
        for vuln in vulnerability_nodes[:5]:  # Limit to prevent explosion
            for exploit in exploit_nodes[:5]:
                try:
                    if self.nx.has_path(self.graph, vuln, exploit):
                        path = self.nx.shortest_path(self.graph, vuln, exploit)
                        paths.append(path)
                except:
                    continue
        
        return {
            'attack_paths': paths[:10],  # Top 10 paths
            'avg_path_length': np.mean([len(p) for p in paths]) if paths else 0
        }
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on graph analysis."""
        recommendations = []
        
        if len(self.graph.nodes) < 10:
            recommendations.append({
                'type': 'knowledge_expansion',
                'priority': 'high',
                'description': 'Knowledge graph is sparse. Consider adding more atoms and relationships.'
            })
        
        # Find isolated nodes
        isolated = list(self.nx.isolates(self.graph))
        if isolated:
            recommendations.append({
                'type': 'relationship_building',
                'priority': 'medium',
                'description': f'{len(isolated)} atoms have no relationships. Consider linking them to related atoms.'
            })
        
        return recommendations
