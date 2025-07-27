#!/usr/bin/env python3
"""
Advanced Machine Learning Models Test Script
Tests ML capabilities for security analysis and threat detection
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xorb_core.ml.advanced_models import (
    MLModelManager, ModelType, NetworkAnomalyDetector, 
    BehavioralAnalysisModel, ThreatClassificationModel, RiskScoringModel,
    demo_ml_models
)
import logging
import numpy as np
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_network_anomaly_detection():
    """Test network anomaly detection model"""
    logger.info("=== Testing Network Anomaly Detection ===")
    
    detector = NetworkAnomalyDetector()
    
    # Generate normal training data
    normal_data = []
    for _ in range(50):
        normal_data.append({
            'packet_count': np.random.normal(1000, 100),
            'byte_count': np.random.normal(50000, 5000),
            'duration': np.random.normal(60, 10),
            'packets_per_second': np.random.normal(16, 2),
            'bytes_per_second': np.random.normal(800, 100),
            'protocols': {'tcp': 0.7, 'udp': 0.2, 'icmp': 0.05, 'other': 0.05},
            'ports': {
                'unique_src_ports': list(range(np.random.randint(3, 8))),
                'unique_dst_ports': list(range(np.random.randint(2, 5))),
                'well_known_ports': np.random.randint(1, 4),
                'high_ports': np.random.randint(2, 6)
            },
            'packet_sizes': [64, 128, 256, 512, 1024],
            'inter_arrival_times': [0.01, 0.02, 0.015, 0.025, 0.018]
        })
    
    # Train model
    training_result = detector.train(normal_data)
    assert training_result["samples_trained"] == 50, "Should train on 50 samples"
    assert detector.is_trained, "Model should be trained"
    
    # Test normal traffic
    normal_traffic = normal_data[0]
    normal_result = detector.predict(normal_traffic)
    assert normal_result.prediction in [0, 1], "Prediction should be 0 or 1"
    assert 0 <= normal_result.confidence <= 1, "Confidence should be between 0 and 1"
    
    # Test anomalous traffic
    anomalous_traffic = {
        'packet_count': 50000,  # Very high
        'byte_count': 1000000,  # Very high
        'duration': 600,        # Very long
        'packets_per_second': 200,  # Very high rate
        'bytes_per_second': 10000,
        'protocols': {'tcp': 0.1, 'udp': 0.1, 'icmp': 0.8, 'other': 0.0},  # Unusual
        'ports': {
            'unique_src_ports': list(range(100)),  # Many ports
            'unique_dst_ports': list(range(50)),
            'well_known_ports': 0,
            'high_ports': 150
        },
        'packet_sizes': [1500] * 10,
        'inter_arrival_times': [0.001] * 10
    }
    
    anomaly_result = detector.predict(anomalous_traffic)
    assert anomaly_result.model_version == "network_anomaly_v1.0", "Should have correct version"
    
    logger.info(f"✅ Normal traffic prediction: {normal_result.prediction} (confidence: {normal_result.confidence:.3f})")
    logger.info(f"✅ Anomaly prediction: {anomaly_result.prediction} (confidence: {anomaly_result.confidence:.3f})")
    logger.info("✅ Network anomaly detection test passed")

async def test_behavioral_analysis():
    """Test behavioral analysis model"""
    logger.info("=== Testing Behavioral Analysis ===")
    
    analyzer = BehavioralAnalysisModel()
    
    # Generate user activity data
    user_activities = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(20):
        activity_time = base_time + timedelta(days=i, hours=np.random.randint(8, 18))
        user_activities.append({
            'activity_times': [activity_time.isoformat()],
            'login_count': np.random.randint(1, 5),
            'failed_login_count': np.random.randint(0, 2),
            'file_access_count': np.random.randint(10, 50),
            'command_count': np.random.randint(20, 100),
            'network_connection_count': np.random.randint(5, 20),
            'resource_usage': {
                'avg_cpu_usage': np.random.uniform(10, 30),
                'max_cpu_usage': np.random.uniform(40, 70),
                'avg_memory_usage': np.random.uniform(20, 40),
                'max_memory_usage': np.random.uniform(50, 80),
                'disk_io_ops': np.random.randint(100, 500)
            },
            'access_patterns': {
                'unique_files': [f"file_{j}" for j in range(np.random.randint(5, 15))],
                'unique_directories': [f"dir_{j}" for j in range(np.random.randint(2, 8))],
                'sensitive_file_access': np.random.randint(0, 3),
                'unique_ip_addresses': [f"192.168.1.{j}" for j in range(np.random.randint(1, 5))],
                'privilege_escalation_attempts': np.random.randint(0, 1)
            }
        })
    
    # Build user profile
    user_id = "test_user_001"
    profile = analyzer.build_user_profile(user_id, user_activities)
    
    assert profile["user_id"] == user_id, "Profile should have correct user ID"
    assert profile["total_activities"] == 20, "Profile should record correct activity count"
    assert len(profile["feature_means"]) > 0, "Profile should have feature statistics"
    
    # Test normal behavior detection
    normal_activity = user_activities[0]  # Use first activity as baseline
    normal_result = analyzer.detect_anomalous_behavior(user_id, normal_activity)
    
    # Test anomalous behavior
    anomalous_activity = {
        'activity_times': [(datetime.now().replace(hour=2)).isoformat()],  # 2 AM unusual
        'login_count': 20,  # Very high
        'failed_login_count': 10,  # Many failures
        'file_access_count': 500,  # Very high
        'command_count': 1000,  # Very high
        'network_connection_count': 100,  # Very high
        'resource_usage': {
            'avg_cpu_usage': 95,  # Very high
            'max_cpu_usage': 100,
            'avg_memory_usage': 90,  # Very high
            'max_memory_usage': 100,
            'disk_io_ops': 5000  # Very high
        },
        'access_patterns': {
            'unique_files': [f"sensitive_file_{j}" for j in range(50)],  # Many files
            'unique_directories': [f"system_dir_{j}" for j in range(20)],  # Many dirs
            'sensitive_file_access': 15,  # High sensitive access
            'unique_ip_addresses': [f"10.0.0.{j}" for j in range(20)],  # Many IPs
            'privilege_escalation_attempts': 5  # Many escalation attempts
        }
    }
    
    anomaly_result = analyzer.detect_anomalous_behavior(user_id, anomalous_activity)
    
    assert normal_result.model_version == "behavioral_v1.0", "Should have correct version"
    assert anomaly_result.model_version == "behavioral_v1.0", "Should have correct version"
    
    logger.info(f"✅ Normal behavior: {normal_result.prediction} (confidence: {normal_result.confidence:.3f})")
    logger.info(f"✅ Anomalous behavior: {anomaly_result.prediction} (confidence: {anomaly_result.confidence:.3f})")
    logger.info("✅ Behavioral analysis test passed")

async def test_threat_classification():
    """Test threat classification model"""
    logger.info("=== Testing Threat Classification ===")
    
    classifier = ThreatClassificationModel()
    
    # Generate training data for different threat types
    training_data = []
    labels = []
    
    threat_templates = {
        'malware': {
            'severity_level': (7, 10),
            'duration_minutes': (30, 300),
            'affected_systems_count': (1, 10),
            'data_exfiltration_attempted': 1,
            'privilege_escalation': 1,
            'attack_vectors': {'network_based': 1, 'email_based': 0, 'web_based': 1, 'physical_access': 0, 'social_engineering': 0},
            'technical_indicators': {'malware_detected': 1, 'suspicious_processes': 1, 'network_anomalies': 0, 'file_modifications': 1, 'registry_changes': 1},
            'mitre_tactics': {'initial_access': 1, 'execution': 1, 'persistence': 1, 'privilege_escalation': 1, 'defense_evasion': 1, 'credential_access': 0, 'discovery': 0, 'lateral_movement': 0, 'collection': 0, 'exfiltration': 1, 'impact': 1}
        },
        'phishing': {
            'severity_level': (4, 8),
            'duration_minutes': (5, 60),
            'affected_systems_count': (1, 3),
            'data_exfiltration_attempted': 0,
            'privilege_escalation': 0,
            'attack_vectors': {'network_based': 0, 'email_based': 1, 'web_based': 1, 'physical_access': 0, 'social_engineering': 1},
            'technical_indicators': {'malware_detected': 0, 'suspicious_processes': 0, 'network_anomalies': 0, 'file_modifications': 0, 'registry_changes': 0},
            'mitre_tactics': {'initial_access': 1, 'execution': 0, 'persistence': 0, 'privilege_escalation': 0, 'defense_evasion': 0, 'credential_access': 1, 'discovery': 0, 'lateral_movement': 0, 'collection': 1, 'exfiltration': 0, 'impact': 0}
        },
        'ddos': {
            'severity_level': (6, 9),
            'duration_minutes': (10, 240),
            'affected_systems_count': (1, 5),
            'data_exfiltration_attempted': 0,
            'privilege_escalation': 0,
            'attack_vectors': {'network_based': 1, 'email_based': 0, 'web_based': 0, 'physical_access': 0, 'social_engineering': 0},
            'technical_indicators': {'malware_detected': 0, 'suspicious_processes': 0, 'network_anomalies': 1, 'file_modifications': 0, 'registry_changes': 0},
            'mitre_tactics': {'initial_access': 0, 'execution': 0, 'persistence': 0, 'privilege_escalation': 0, 'defense_evasion': 0, 'credential_access': 0, 'discovery': 0, 'lateral_movement': 0, 'collection': 0, 'exfiltration': 0, 'impact': 1}
        }
    }
    
    # Generate training samples
    for threat_type, template in threat_templates.items():
        for _ in range(30):  # 30 samples per type
            incident = {}
            
            for key, value in template.items():
                if isinstance(value, tuple):
                    # Range values
                    incident[key] = np.random.uniform(value[0], value[1])
                elif isinstance(value, dict):
                    # Dictionary values
                    incident[key] = value.copy()
                else:
                    # Fixed values
                    incident[key] = value
            
            training_data.append(incident)
            labels.append(threat_type)
    
    # Train classifier
    training_result = classifier.train(training_data, labels)
    
    assert training_result.accuracy > 0.7, f"Accuracy should be > 0.7, got {training_result.accuracy}"
    assert classifier.is_trained, "Classifier should be trained"
    assert len(classifier.threat_classes) == 3, "Should have 3 threat classes"
    
    # Test classification
    test_malware = {
        'severity_level': 8,
        'duration_minutes': 120,
        'affected_systems_count': 5,
        'data_exfiltration_attempted': 1,
        'privilege_escalation': 1,
        'attack_vectors': {'network_based': 1, 'email_based': 0, 'web_based': 1, 'physical_access': 0, 'social_engineering': 0},
        'technical_indicators': {'malware_detected': 1, 'suspicious_processes': 1, 'network_anomalies': 0, 'file_modifications': 1, 'registry_changes': 1},
        'mitre_tactics': {'initial_access': 1, 'execution': 1, 'persistence': 1, 'privilege_escalation': 1, 'defense_evasion': 1, 'credential_access': 0, 'discovery': 0, 'lateral_movement': 0, 'collection': 0, 'exfiltration': 1, 'impact': 1}
    }
    
    classification_result = classifier.predict(test_malware)
    
    assert classification_result.prediction in classifier.threat_classes, "Prediction should be valid threat class"
    assert 0 <= classification_result.confidence <= 1, "Confidence should be between 0 and 1"
    
    logger.info(f"✅ Training accuracy: {training_result.accuracy:.3f}")
    logger.info(f"✅ Classification result: {classification_result.prediction} (confidence: {classification_result.confidence:.3f})")
    logger.info("✅ Threat classification test passed")

async def test_risk_scoring():
    """Test risk scoring model"""
    logger.info("=== Testing Risk Scoring ===")
    
    risk_model = RiskScoringModel()
    
    # Test low-risk asset
    low_risk_asset = {
        'criticality_level': 0.2,
        'vulnerabilities': [{'cvss_score': 3.1, 'name': 'CVE-2023-0001'}],
        'exposure_level': 0.1,
        'recent_threat_activity': 0.0,
        'security_control_effectiveness': 0.9
    }
    
    low_risk_context = {
        'current_time': datetime.now().replace(hour=14).isoformat(),  # Business hours
        'recent_incidents': [],
        'threat_intel': {'active_campaigns': 0},
        'business_context': {'critical_business_period': False}
    }
    
    low_risk_result = risk_model.calculate_dynamic_risk_score(low_risk_asset, low_risk_context)
    
    # Test high-risk asset
    high_risk_asset = {
        'criticality_level': 0.9,
        'vulnerabilities': [
            {'cvss_score': 9.8, 'name': 'CVE-2023-9999'},
            {'cvss_score': 8.5, 'name': 'CVE-2023-8888'}
        ],
        'exposure_level': 0.8,
        'recent_threat_activity': 0.7,
        'security_control_effectiveness': 0.3
    }
    
    high_risk_context = {
        'current_time': datetime.now().replace(hour=2).isoformat(),  # Off hours
        'recent_incidents': [
            {'severity': 9, 'type': 'apt'},
            {'severity': 8, 'type': 'malware'},
            {'severity': 7, 'type': 'data_breach'}
        ],
        'threat_intel': {'active_campaigns': 5},
        'business_context': {
            'critical_business_period': True,
            'under_regulatory_scrutiny': True,
            'high_public_visibility': True
        }
    }
    
    high_risk_result = risk_model.calculate_dynamic_risk_score(high_risk_asset, high_risk_context)
    
    # Verify risk scoring logic
    assert 0 <= low_risk_result['base_risk_score'] <= 1, "Base risk score should be between 0 and 1"
    assert 0 <= low_risk_result['dynamic_risk_score'] <= 1, "Dynamic risk score should be between 0 and 1"
    assert low_risk_result['dynamic_risk_score'] < high_risk_result['dynamic_risk_score'], "High-risk asset should have higher score"
    
    assert high_risk_result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'VERY_LOW'], "Should have valid risk level"
    assert len(high_risk_result['recommendations']) > 0, "High-risk asset should have recommendations"
    
    logger.info(f"✅ Low risk score: {low_risk_result['dynamic_risk_score']:.3f} ({low_risk_result['risk_level']})")
    logger.info(f"✅ High risk score: {high_risk_result['dynamic_risk_score']:.3f} ({high_risk_result['risk_level']})")
    logger.info(f"✅ Risk factors: {list(high_risk_result['risk_factors'].keys())}")
    logger.info("✅ Risk scoring test passed")

async def test_model_manager():
    """Test ML model manager"""
    logger.info("=== Testing ML Model Manager ===")
    
    model_manager = MLModelManager()
    
    # Test model initialization
    assert ModelType.ANOMALY_DETECTION in model_manager.models, "Should have anomaly detection model"
    assert ModelType.BEHAVIORAL_ANALYSIS in model_manager.models, "Should have behavioral analysis model"
    assert ModelType.THREAT_CLASSIFICATION in model_manager.models, "Should have threat classification model"
    assert ModelType.RISK_SCORING in model_manager.models, "Should have risk scoring model"
    
    # Test model info
    for model_type in model_manager.models.keys():
        info = await model_manager.get_model_info(model_type)
        assert info["model_type"] == model_type.value, f"Model info should have correct type for {model_type}"
        assert "is_trained" in info, f"Model info should have training status for {model_type}"
        assert "model_class" in info, f"Model info should have class name for {model_type}"
        
        logger.info(f"✅ {model_type.value}: {info['model_class']} (trained: {info['is_trained']})")
    
    # Test simple anomaly detection training and prediction
    normal_data = []
    for _ in range(20):
        normal_data.append({
            'packet_count': np.random.normal(1000, 100),
            'byte_count': np.random.normal(50000, 5000),
            'duration': np.random.normal(60, 10),
            'packets_per_second': np.random.normal(16, 2),
            'bytes_per_second': np.random.normal(800, 100),
            'protocols': {'tcp': 0.7, 'udp': 0.2, 'icmp': 0.05, 'other': 0.05},
            'ports': {
                'unique_src_ports': list(range(5)),
                'unique_dst_ports': list(range(3)),
                'well_known_ports': 2,
                'high_ports': 3
            },
            'packet_sizes': [64, 128, 256, 512, 1024],
            'inter_arrival_times': [0.01, 0.02, 0.015, 0.025, 0.018]
        })
    
    # Train through manager
    training_result = await model_manager.train_model(ModelType.ANOMALY_DETECTION, normal_data)
    assert training_result.model_id.startswith("anomaly_detection"), "Should have correct model ID"
    
    # Predict through manager
    test_data = normal_data[0]
    prediction_result = await model_manager.predict(ModelType.ANOMALY_DETECTION, test_data)
    assert prediction_result.model_version == "network_anomaly_v1.0", "Should have correct version"
    
    # Test risk scoring through manager
    asset_data = {
        'criticality_level': 0.5,
        'vulnerabilities': [{'cvss_score': 6.0, 'name': 'CVE-2023-TEST'}],
        'exposure_level': 0.3,
        'recent_threat_activity': 0.2,
        'security_control_effectiveness': 0.8
    }
    
    context_data = {
        'current_time': datetime.now().isoformat(),
        'recent_incidents': [],
        'threat_intel': {'active_campaigns': 1},
        'business_context': {}
    }
    
    risk_result = await model_manager.predict(ModelType.RISK_SCORING, asset_data, context_data=context_data)
    assert 0 <= risk_result.prediction <= 1, "Risk score should be between 0 and 1"
    
    logger.info(f"✅ Anomaly prediction: {prediction_result.prediction} (confidence: {prediction_result.confidence:.3f})")
    logger.info(f"✅ Risk prediction: {risk_result.prediction:.3f}")
    logger.info("✅ Model manager test passed")

async def test_model_persistence():
    """Test model saving and loading"""
    logger.info("=== Testing Model Persistence ===")
    
    # Create temporary model manager
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        model_manager = MLModelManager(temp_dir)
        
        # Train a simple model
        normal_data = []
        for _ in range(10):
            normal_data.append({
                'packet_count': np.random.normal(1000, 100),
                'byte_count': np.random.normal(50000, 5000),
                'duration': np.random.normal(60, 10),
                'packets_per_second': np.random.normal(16, 2),
                'bytes_per_second': np.random.normal(800, 100),
                'protocols': {'tcp': 0.7, 'udp': 0.2, 'icmp': 0.05, 'other': 0.05},
                'ports': {
                    'unique_src_ports': list(range(5)),
                    'unique_dst_ports': list(range(3)),
                    'well_known_ports': 2,
                    'high_ports': 3
                },
                'packet_sizes': [64, 128, 256, 512, 1024],
                'inter_arrival_times': [0.01, 0.02, 0.015, 0.025, 0.018]
            })
        
        await model_manager.train_model(ModelType.ANOMALY_DETECTION, normal_data)
        
        # Model should be trained
        info = await model_manager.get_model_info(ModelType.ANOMALY_DETECTION)
        assert info["is_trained"], "Model should be trained before saving"
        
        # Save model
        await model_manager.save_model(ModelType.ANOMALY_DETECTION)
        
        # Create new manager and load model
        model_manager2 = MLModelManager(temp_dir)
        
        # Before loading, model should not be trained
        info2 = await model_manager2.get_model_info(ModelType.ANOMALY_DETECTION)
        assert not info2["is_trained"], "New model should not be trained initially"
        
        # Load model
        await model_manager2.load_model(ModelType.ANOMALY_DETECTION)
        
        # After loading, model should be trained
        info3 = await model_manager2.get_model_info(ModelType.ANOMALY_DETECTION)
        assert info3["is_trained"], "Loaded model should be trained"
        
        # Test prediction with loaded model
        test_data = normal_data[0]
        result = await model_manager2.predict(ModelType.ANOMALY_DETECTION, test_data)
        assert result.prediction in [0, 1], "Loaded model should work for predictions"
    
    logger.info("✅ Model persistence test passed")

async def test_feature_extraction():
    """Test feature extraction methods"""
    logger.info("=== Testing Feature Extraction ===")
    
    # Test network feature extraction
    detector = NetworkAnomalyDetector()
    
    network_data = {
        'packet_count': 1000,
        'byte_count': 50000,
        'duration': 60,
        'packets_per_second': 16.67,
        'bytes_per_second': 833.33,
        'protocols': {'tcp': 0.7, 'udp': 0.2, 'icmp': 0.05, 'other': 0.05},
        'ports': {
            'unique_src_ports': list(range(5)),
            'unique_dst_ports': list(range(3)),
            'well_known_ports': 2,
            'high_ports': 3
        },
        'packet_sizes': [64, 128, 256, 512, 1024],
        'inter_arrival_times': [0.01, 0.02, 0.015, 0.025, 0.018]
    }
    
    features = detector.extract_network_features(network_data)
    assert len(features) == 21, f"Should extract 21 features, got {len(features)}"
    assert all(isinstance(f, (int, float)) for f in features), "All features should be numeric"
    
    # Test behavioral feature extraction
    analyzer = BehavioralAnalysisModel()
    
    user_activity = {
        'activity_times': [datetime.now().isoformat()],
        'login_count': 3,
        'failed_login_count': 1,
        'file_access_count': 25,
        'command_count': 50,
        'network_connection_count': 10,
        'resource_usage': {
            'avg_cpu_usage': 15.5,
            'max_cpu_usage': 45.2,
            'avg_memory_usage': 25.8,
            'max_memory_usage': 62.1,
            'disk_io_ops': 150
        },
        'access_patterns': {
            'unique_files': ['file1', 'file2', 'file3'],
            'unique_directories': ['dir1', 'dir2'],
            'sensitive_file_access': 1,
            'unique_ip_addresses': ['192.168.1.100', '192.168.1.101'],
            'privilege_escalation_attempts': 0
        }
    }
    
    behavioral_features = analyzer.extract_behavioral_features(user_activity)
    assert len(behavioral_features) == 20, f"Should extract 20 behavioral features, got {len(behavioral_features)}"
    assert all(isinstance(f, (int, float)) for f in behavioral_features), "All features should be numeric"
    
    # Test threat feature extraction
    classifier = ThreatClassificationModel()
    
    incident_data = {
        'severity_level': 8,
        'duration_minutes': 120,
        'affected_systems_count': 3,
        'data_exfiltration_attempted': 1,
        'privilege_escalation': 1,
        'attack_vectors': {'network_based': 1, 'email_based': 0, 'web_based': 1, 'physical_access': 0, 'social_engineering': 0},
        'technical_indicators': {'malware_detected': 1, 'suspicious_processes': 1, 'network_anomalies': 0, 'file_modifications': 1, 'registry_changes': 1},
        'mitre_tactics': {'initial_access': 1, 'execution': 1, 'persistence': 1, 'privilege_escalation': 1, 'defense_evasion': 1, 'credential_access': 0, 'discovery': 0, 'lateral_movement': 0, 'collection': 0, 'exfiltration': 1, 'impact': 1}
    }
    
    threat_features = classifier.extract_threat_features(incident_data)
    assert len(threat_features) == 26, f"Should extract 26 threat features, got {len(threat_features)}"
    assert all(isinstance(f, (int, float)) for f in threat_features), "All features should be numeric"
    
    logger.info(f"✅ Network features: {len(features)} extracted")
    logger.info(f"✅ Behavioral features: {len(behavioral_features)} extracted")
    logger.info(f"✅ Threat features: {len(threat_features)} extracted")
    logger.info("✅ Feature extraction test passed")

async def main():
    """Run all ML model tests"""
    logger.info("Starting Advanced Machine Learning Models Tests")
    logger.info("=" * 70)
    
    try:
        # Run comprehensive test suite
        await test_network_anomaly_detection()
        await test_behavioral_analysis()
        await test_threat_classification()
        await test_risk_scoring()
        await test_model_manager()
        await test_model_persistence()
        await test_feature_extraction()
        
        # Run the comprehensive demo
        logger.info("=== Running ML Models Demo ===")
        await demo_ml_models()
        
        logger.info("=" * 70)
        logger.info("🎉 All ML model tests passed!")
        
        logger.info("\n📊 Test Summary:")
        logger.info("  ✅ Network Anomaly Detection - Isolation Forest based anomaly detection")
        logger.info("  ✅ Behavioral Analysis - User behavior profiling and anomaly detection")
        logger.info("  ✅ Threat Classification - Multi-class threat categorization")
        logger.info("  ✅ Risk Scoring - Dynamic risk assessment with contextual factors")
        logger.info("  ✅ Model Management - Centralized ML model lifecycle management")
        logger.info("  ✅ Model Persistence - Save/load functionality for trained models")
        logger.info("  ✅ Feature Extraction - Comprehensive feature engineering")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())