"""
Behavioral Analytics Engine for PTAAS

This module implements machine learning-based behavioral analysis for threat detection.
"""
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)

class BehavioralProfile:
    """Represents a behavioral profile for a user or system entity."""

    def __init__(self, entity_id: str, profile_type: str):
        """
        Initialize a behavioral profile.

        Args:
            entity_id: Unique identifier for the entity
            profile_type: Type of profile (user, system, etc.)
        """
        self.entity_id = entity_id
        self.profile_type = profile_type
        self.features = {}
        self.timestamps = []
        self.model = None
        self.last_update = datetime.utcnow()
        self.scaler = StandardScaler()

    def add_observation(self, features: Dict[str, float], timestamp: Optional[datetime] = None):
        """
        Add a new observation to the profile.

        Args:
            features: Dictionary of feature values
            timestamp: Optional timestamp for the observation
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        self.timestamps.append(timestamp)

        # Update features
        for feature_name, value in features.items():
            if feature_name not in self.features:
                self.features[feature_name] = []
            self.features[feature_name].append(value)

    def create_model(self) -> None:
        """Create or update the behavioral model based on collected observations."""
        if len(self.timestamps) < 2:
            logger.warning(f"Not enough observations to create model for {self.entity_id}")
            return

        # Create feature matrix
        feature_matrix = np.array([
            [self.features[feature][i] for feature in sorted(self.features.keys())]
            for i in range(len(self.timestamps))
        ])

        # Scale features
        scaled_features = self.scaler.fit_transform(feature_matrix)

        # Create anomaly detection model using Isolation Forest
        self.model = IsolationForest(n_estimators=100, contamination=0.1, behaviour='new')
        self.model.fit(scaled_features)

        # Update last update time
        self.last_update = datetime.utcnow()

    def calculate_risk_score(self, current_time: Optional[datetime] = None) -> float:
        """
        Calculate a risk score based on behavioral anomalies.

        Args:
            current_time: Optional current time for decay calculation

        Returns:
            Risk score between 0 and 1
        """
        if current_time is None:
            current_time = datetime.utcnow()

        if self.model is None:
            logger.warning(f"Model not trained for {self.entity_id}")
            return 0.0

        # Get latest observation
        latest_features = np.array([[
            self.features[feature][-1] for feature in sorted(self.features.keys())
        ]])

        # Scale features
        scaled_features = self.scaler.transform(latest_features)

        # Get anomaly score
        anomaly_score = float(self.model.score_samples(scaled_features)[0])

        # Convert to risk score (lower score means more anomalous)
        risk_score = max(0.0, 1.0 + anomaly_score)  # Scores closer to 0 are more anomalous

        # Apply time decay (older profiles have less relevance)
        time_diff = (current_time - self.last_update).total_seconds() / 3600  # hours
        decay_factor = np.exp(-time_diff / 24)  # 24 hour half-life

        return risk_score * decay_factor

    def detect_pattern(self, pattern_window: int = 24) -> Dict[str, Any]:
        """
        Detect patterns in behavioral data.

        Args:
            pattern_window: Number of hours to consider for pattern detection

        Returns:
            Dictionary containing detected patterns
        """
        patterns = {
            "anomalies": [],
            "trends": [],
            "recurring": [],
            "risk_level": "low"
        }

        if len(self.timestamps) < 2:
            return patterns

        # Find recent observations
        current_time = datetime.utcnow()
        recent_indices = [
            i for i, ts in enumerate(self.timestamps)
            if (current_time - ts).total_seconds() <= pattern_window * 3600
        ]

        if len(recent_indices) < 2:
            return patterns

        # Detect anomalies using DBSCAN clustering
        feature_matrix = np.array([
            [self.features[feature][i] for feature in sorted(self.features.keys())]
            for i in recent_indices
        ])

        clusters = DBSCAN(eps=0.5, min_samples=2).fit_predict(feature_matrix)

        # Identify outliers
        outliers = np.where(clusters == -1)[0]
        if len(outliers) > 0:
            patterns["anomalies"].append({
                "count": len(outliers),
                "percentage": len(outliers) / len(recent_indices) * 100,
                "indices": [recent_indices[i] for i in outliers]
            })

        # Detect trends for numerical features
        for feature in sorted(self.features.keys()):
            values = [self.features[feature][i] for i in recent_indices]
            if not isinstance(values[0], (int, float)):
                continue

            # Calculate linear regression slope
            x = np.array(range(len(values))).reshape(-1, 1)
            y = np.array(values)
            slope = np.polyfit(x.ravel(), y, 1)[0] if len(values) > 1 else 0

            patterns["trends"].append({
                "feature": feature,
                "slope": float(slope),
                "trend_type": "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
            })

        # Detect recurring patterns
        if len(self.timestamps) > pattern_window * 2:
            # Compare recent behavior with historical data
            historical_indices = [
                i for i, ts in enumerate(self.timestamps)
                if (current_time - ts).total_seconds() > pattern_window * 3600
            ]

            if len(historical_indices) > 0:
                historical_matrix = np.array([
                    [self.features[feature][i] for feature in sorted(self.features.keys())]
                    for i in historical_indices
                ])

                recent_center = np.mean(feature_matrix, axis=0)
                historical_center = np.mean(historical_matrix, axis=0)

                distance = pairwise_distances([recent_center], [historical_center])[0][0]

                patterns["recurring"].append({
                    "similarity": float(1 / (1 + distance)),
                    "pattern_changed": distance > 0.5
                })

        # Determine risk level based on patterns
        anomaly_count = sum(a["count"] for a in patterns["anomalies"])
        if anomaly_count > 3:
            patterns["risk_level"] = "high"
        elif anomaly_count > 1:
            patterns["risk_level"] = "medium"
        else:
            patterns["risk_level"] = "low"

        return patterns


class BehavioralAnalyticsEngine:
    """Main engine for behavioral analytics and anomaly detection."""

    def __init__(self):
        """Initialize the behavioral analytics engine."""
        self.profiles = {}  # Dictionary to store behavioral profiles
        self.default_profile_window = 24 * 7  # 7 days

    def create_profile(self, entity_id: str, profile_type: str) -> BehavioralProfile:
        """
        Create a new behavioral profile.

        Args:
            entity_id: Unique identifier for the entity
            profile_type: Type of profile (user, system, etc.)

        Returns:
            Created BehavioralProfile object
        """
        if entity_id in self.profiles:
            raise ValueError(f"Profile already exists for {entity_id}")

        profile = BehavioralProfile(entity_id, profile_type)
        self.profiles[entity_id] = profile
        return profile

    def get_profile(self, entity_id: str) -> Optional[BehavioralProfile]:
        """
        Get an existing behavioral profile.

        Args:
            entity_id: Unique identifier for the entity

        Returns:
            BehavioralProfile object or None if not found
        """
        return self.profiles.get(entity_id)

    def update_profile(self, entity_id: str, features: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """
        Update a behavioral profile with new observations.

        Args:
            entity_id: Unique identifier for the entity
            features: Dictionary of feature values
            timestamp: Optional timestamp for the observation
        """
        profile = self.get_profile(entity_id)
        if profile is None:
            profile = self.create_profile(entity_id, "user")  # Default to user type

        profile.add_observation(features, timestamp)

        # Prune old observations if needed
        self._prune_profile(profile)

        # Recreate model if needed
        if len(profile.timestamps) >= 2:
            profile.create_model()

    def _prune_profile(self, profile: BehavioralProfile) -> None:
        """
        Prune old observations from a profile.

        Args:
            profile: BehavioralProfile to prune
        """
        current_time = datetime.utcnow()

        # Remove observations older than the window
        cutoff_time = current_time - timedelta(hours=self.default_profile_window)
        valid_indices = [
            i for i, ts in enumerate(profile.timestamps)
            if ts >= cutoff_time
        ]

        # Keep only valid indices
        profile.timestamps = [profile.timestamps[i] for i in valid_indices]

        for feature in profile.features:
            profile.features[feature] = [profile.features[feature][i] for i in valid_indices]

    def analyze_behavior(self, entity_id: str, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze behavior and detect anomalies.

        Args:
            entity_id: Unique identifier for the entity
            features: Dictionary of current feature values

        Returns:
            Dictionary containing analysis results
        """
        # Get or create profile
        profile = self.get_profile(entity_id)
        if profile is None:
            profile = self.create_profile(entity_id, "user")  # Default to user type

        # Add current observation
        profile.add_observation(features)

        # Prune old observations
        self._prune_profile(profile)

        # Create model if needed
        if len(profile.timestamps) >= 2:
            profile.create_model()

        # Calculate risk score
        risk_score = profile.calculate_risk_score()

        # Detect patterns
        patterns = profile.detect_pattern()

        # Create analysis result
        analysis = {
            "entity_id": entity_id,
            "risk_score": risk_score,
            "risk_level": self._determine_risk_level(risk_score),
            "patterns": patterns,
            "last_update": profile.last_update.isoformat(),
            "observation_count": len(profile.timestamps)
        }

        return analysis

    def _determine_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level based on risk score.

        Args:
            risk_score: Calculated risk score (0-1)

        Returns:
            Risk level (low/medium/high/critical)
        """
        if risk_score > 0.8:
            return "critical"
        elif risk_score > 0.6:
            return "high"
        elif risk_score > 0.4:
            return "medium"
        else:
            return "low"

    def generate_alert(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate an alert if risk level is above threshold.

        Args:
            analysis: Analysis results from analyze_behavior

        Returns:
            Alert dictionary or None if no alert needed
        """
        risk_level = analysis["risk_level"]

        if risk_level in ["high", "critical"]:
            alert = {
                "timestamp": datetime.utcnow().isoformat(),
                "entity_id": analysis["entity_id"],
                "risk_level": risk_level,
                "risk_score": analysis["risk_score"],
                "patterns": analysis["patterns"],
                "description": self._generate_alert_description(analysis)
            }

            return alert

        return None

    def _generate_alert_description(self, analysis: Dict[str, Any]) -> str:
        """
        Generate a description for an alert.

        Args:
            analysis: Analysis results

        Returns:
            Alert description string
        """
        risk_level = analysis["risk_level"]
        patterns = analysis["patterns"]

        description = f"Behavioral anomaly detected for entity {analysis['entity_id']}\n"
        description += f"Risk Level: {risk_level.upper()}\n\n"

        # Add pattern details
        if patterns["risk_level"] != "low":
            description += "Detected Behavioral Patterns:\n"

            # Add anomalies
            if patterns["anomalies"]:
                description += f"- Detected {patterns['anomalies'][0]['count']} anomalies in recent behavior\n"

            # Add trends
            for trend in patterns["trends"]:
                if trend["trend_type"] != "stable":
                    description += f"- Feature '{trend['feature']}' showing {trend['trend_type']} trend\n"

            # Add recurring patterns
            if patterns["recurring"] and patterns["recurring"][0]["pattern_changed"]:
                description += "- Significant change in recurring behavior patterns\n"

        description += "\nRecommend immediate investigation and verification of this behavior."

        return description


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        # Initialize engine
        engine = BehavioralAnalyticsEngine()

        # Example user ID
        user_id = "user_12345"

        # Example features (could be login time, location, activity patterns, etc.)
        features = {
            "login_hour": 2,  # Late night login
            "location_entropy": 0.9,  # High entropy indicates many different locations
            "access_pattern_deviation": 0.7,  # Deviation from normal access patterns
            "failed_login_attempts": 5
        }

        # Analyze behavior
        analysis = engine.analyze_behavior(user_id, features)

        # Print analysis results
        print(f"Entity ID: {analysis['entity_id']}")
        print(f"Risk Score: {analysis['risk_score']:.2f}")
        print(f"Risk Level: {analysis['risk_level']}")
        print(f"Observation Count: {analysis['observation_count']}")

        # Generate alert if needed
        alert = engine.generate_alert(analysis)
        if alert:
            print("\nSecurity Alert:")
            print(alert["description"])

    asyncio.run(main())
