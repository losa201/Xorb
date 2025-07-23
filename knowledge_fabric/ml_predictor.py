#!/usr/bin/env python3

import logging
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

from .atom import KnowledgeAtom, AtomType, ConfidenceLevel


@dataclass
class PredictionFeatures:
    age_days: float
    usage_frequency: float
    success_rate: float
    source_reliability: float
    validation_score: float
    relationship_count: float
    content_complexity: float
    atom_type_encoding: List[float]


class ConfidencePredictor:
    def __init__(self, model_path: str = "./models/confidence_predictor.pkl"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.feature_weights = {
            'age_days': -0.05,
            'usage_frequency': 0.3,
            'success_rate': 0.4,
            'source_reliability': 0.2,
            'validation_score': 0.35,
            'relationship_count': 0.1,
            'content_complexity': 0.05
        }
        
        self.atom_type_weights = {
            AtomType.VULNERABILITY: 0.8,
            AtomType.EXPLOIT: 0.9,
            AtomType.TECHNIQUE: 0.7,
            AtomType.PAYLOAD: 0.6,
            AtomType.TARGET_INFO: 0.4,
            AtomType.INTELLIGENCE: 0.5,
            AtomType.DEFENSIVE: 0.7
        }
        
        self.training_data: List[Tuple[PredictionFeatures, float]] = []
        self.model = None
        self.is_trained = False
        
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        if self.model_path.exists():
            await self._load_model()
        else:
            await self._initialize_simple_model()
        
        self.logger.info("Confidence predictor initialized")

    async def predict_confidence(self, atom: KnowledgeAtom) -> float:
        features = await self._extract_features(atom)
        
        if self.is_trained and self.model:
            try:
                confidence = self._predict_with_model(features)
            except Exception as e:
                self.logger.warning(f"Model prediction failed, using heuristic: {e}")
                confidence = self._predict_heuristic(features)
        else:
            confidence = self._predict_heuristic(features)
        
        return max(0.0, min(1.0, confidence))

    async def predict_success_probability(self, atom: KnowledgeAtom, context: Dict[str, Any] = None) -> float:
        base_confidence = await self.predict_confidence(atom)
        
        context_multiplier = 1.0
        if context:
            target_type = context.get('target_type', 'unknown')
            campaign_type = context.get('campaign_type', 'unknown')
            
            if atom.atom_type == AtomType.VULNERABILITY:
                if target_type in ['web_app', 'api']:
                    context_multiplier = 1.2
                elif target_type in ['network', 'infrastructure']:
                    context_multiplier = 0.8
            
            elif atom.atom_type == AtomType.EXPLOIT:
                if campaign_type == 'automated':
                    context_multiplier = atom.success_rate * 1.1
                else:
                    context_multiplier = 0.9
        
        success_probability = base_confidence * context_multiplier
        
        if atom.usage_count > 0:
            historical_factor = (atom.success_rate * 0.7) + (success_probability * 0.3)
            success_probability = historical_factor
        
        return max(0.0, min(1.0, success_probability))

    async def train_on_atom(self, atom: KnowledgeAtom, actual_outcome: Optional[float] = None):
        features = await self._extract_features(atom)
        target_confidence = actual_outcome if actual_outcome is not None else atom.confidence
        
        self.training_data.append((features, target_confidence))
        
        if len(self.training_data) >= 100:
            await self._retrain_model()
        
        if len(self.training_data) % 10 == 0:
            await self._update_feature_weights(features, target_confidence)

    async def evaluate_prediction_accuracy(self, test_atoms: List[KnowledgeAtom]) -> Dict[str, float]:
        if not test_atoms:
            return {"error": "No test atoms provided"}
        
        predictions = []
        actuals = []
        
        for atom in test_atoms:
            predicted = await self.predict_confidence(atom)
            actual = atom.confidence
            
            predictions.append(predicted)
            actuals.append(actual)
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0.0
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "correlation": float(correlation),
            "test_count": len(test_atoms)
        }

    async def get_confidence_trends(self, atoms: List[KnowledgeAtom]) -> Dict[str, Any]:
        if not atoms:
            return {}
        
        trends = {
            "by_type": {},
            "by_age": {},
            "by_usage": {},
            "overall_distribution": {level.value: 0 for level in ConfidenceLevel}
        }
        
        for atom in atoms:
            # By type
            atom_type = atom.atom_type.value
            if atom_type not in trends["by_type"]:
                trends["by_type"][atom_type] = []
            trends["by_type"][atom_type].append(atom.confidence)
            
            # By age
            age_days = (datetime.utcnow() - atom.created_at).days
            age_bucket = f"{(age_days // 7) * 7}-{(age_days // 7 + 1) * 7} days"
            if age_bucket not in trends["by_age"]:
                trends["by_age"][age_bucket] = []
            trends["by_age"][age_bucket].append(atom.confidence)
            
            # By usage
            usage_bucket = "high" if atom.usage_count > 10 else "medium" if atom.usage_count > 2 else "low"
            if usage_bucket not in trends["by_usage"]:
                trends["by_usage"][usage_bucket] = []
            trends["by_usage"][usage_bucket].append(atom.confidence)
            
            # Overall distribution
            conf_level = atom.confidence_level.value
            trends["overall_distribution"][conf_level] += 1
        
        # Calculate averages
        for category in ["by_type", "by_age", "by_usage"]:
            for key, values in trends[category].items():
                trends[category][key] = {
                    "average": np.mean(values),
                    "count": len(values),
                    "std": np.std(values)
                }
        
        return trends

    async def recommend_confidence_improvements(self, atom: KnowledgeAtom) -> List[Dict[str, Any]]:
        recommendations = []
        
        if atom.confidence < 0.5:
            recommendations.append({
                "action": "add_validation",
                "description": "Add validation from multiple sources",
                "expected_improvement": 0.2,
                "priority": "high"
            })
        
        if len(atom.sources) < 2:
            recommendations.append({
                "action": "diversify_sources", 
                "description": "Collect information from additional sources",
                "expected_improvement": 0.15,
                "priority": "medium"
            })
        
        if atom.usage_count == 0:
            recommendations.append({
                "action": "practical_testing",
                "description": "Test in controlled environment to gather usage data",
                "expected_improvement": 0.1,
                "priority": "low"
            })
        
        if len(atom.related_atoms) == 0:
            recommendations.append({
                "action": "establish_relationships",
                "description": "Link to related atoms to improve context",
                "expected_improvement": 0.05,
                "priority": "low"
            })
        
        return recommendations

    async def _extract_features(self, atom: KnowledgeAtom) -> PredictionFeatures:
        age_days = (datetime.utcnow() - atom.created_at).days
        
        usage_frequency = atom.usage_count / max(1, age_days) if age_days > 0 else 0
        
        source_reliability = np.mean([s.reliability_score for s in atom.sources]) if atom.sources else 0.5
        
        validation_score = len([v for v in atom.validation_results if v.is_valid]) / max(1, len(atom.validation_results)) if atom.validation_results else 0.5
        
        content_complexity = len(str(atom.content)) / 1000.0  # Normalized by 1000 chars
        
        # Atom type encoding (one-hot-like)
        atom_type_encoding = [0.0] * len(AtomType)
        for i, at in enumerate(AtomType):
            if at == atom.atom_type:
                atom_type_encoding[i] = 1.0
                break
        
        return PredictionFeatures(
            age_days=age_days,
            usage_frequency=usage_frequency,
            success_rate=atom.success_rate,
            source_reliability=source_reliability,
            validation_score=validation_score,
            relationship_count=len(atom.related_atoms),
            content_complexity=content_complexity,
            atom_type_encoding=atom_type_encoding
        )

    def _predict_heuristic(self, features: PredictionFeatures) -> float:
        confidence = 0.5  # Base confidence
        
        # Apply feature weights
        confidence += self.feature_weights['age_days'] * min(features.age_days / 30.0, 1.0)
        confidence += self.feature_weights['usage_frequency'] * min(features.usage_frequency, 1.0)
        confidence += self.feature_weights['success_rate'] * features.success_rate
        confidence += self.feature_weights['source_reliability'] * features.source_reliability
        confidence += self.feature_weights['validation_score'] * features.validation_score
        confidence += self.feature_weights['relationship_count'] * min(features.relationship_count / 10.0, 1.0)
        confidence += self.feature_weights['content_complexity'] * min(features.content_complexity, 1.0)
        
        # Apply atom type weight
        atom_type_weight = 0.0
        for i, weight in enumerate(self.atom_type_weights.values()):
            if i < len(features.atom_type_encoding) and features.atom_type_encoding[i] > 0:
                atom_type_weight = weight
                break
        
        confidence *= atom_type_weight
        
        return confidence

    def _predict_with_model(self, features: PredictionFeatures) -> float:
        # Placeholder for actual ML model prediction
        # In a real implementation, this would use scikit-learn, TensorFlow, etc.
        return self._predict_heuristic(features)

    async def _retrain_model(self):
        if len(self.training_data) < 10:
            return
        
        self.logger.info(f"Retraining model with {len(self.training_data)} samples")
        
        # Placeholder for actual model training
        # In practice, this would train a regression model
        self.is_trained = True
        await self._save_model()

    async def _update_feature_weights(self, features: PredictionFeatures, target: float):
        learning_rate = 0.01
        predicted = self._predict_heuristic(features)
        error = target - predicted
        
        # Simple gradient descent-like updates
        self.feature_weights['usage_frequency'] += learning_rate * error * features.usage_frequency
        self.feature_weights['success_rate'] += learning_rate * error * features.success_rate
        self.feature_weights['source_reliability'] += learning_rate * error * features.source_reliability
        self.feature_weights['validation_score'] += learning_rate * error * features.validation_score

    async def _save_model(self):
        try:
            model_data = {
                'feature_weights': self.feature_weights,
                'atom_type_weights': self.atom_type_weights,
                'training_data': self.training_data[-1000:],  # Keep last 1000 samples
                'is_trained': self.is_trained
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    async def _load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.feature_weights = model_data.get('feature_weights', self.feature_weights)
            self.atom_type_weights = model_data.get('atom_type_weights', self.atom_type_weights)
            self.training_data = model_data.get('training_data', [])
            self.is_trained = model_data.get('is_trained', False)
            
            self.logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            await self._initialize_simple_model()

    async def _initialize_simple_model(self):
        self.is_trained = False
        self.training_data = []
        self.logger.info("Initialized simple heuristic model")