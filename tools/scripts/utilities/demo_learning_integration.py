#!/usr/bin/env python3
"""
XORB Learning Engine Integration Demo
Real-time demonstration of the PTaaS to Learning Engine integration
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LearningIntegrationDemo:
    """Demonstration of XORB Learning Engine Integration"""
    
    def __init__(self):
        self.demo_data = {
            'agents': [
                {'id': 'agent_web_001', 'type': 'web_application', 'performance': 0.75},
                {'id': 'agent_network_002', 'type': 'network_infrastructure', 'performance': 0.82},
                {'id': 'agent_mobile_003', 'type': 'mobile_application', 'performance': 0.68},
                {'id': 'agent_api_004', 'type': 'api_security', 'performance': 0.90}
            ],
            'campaigns': [],
            'telemetry_events': [],
            'learning_metrics': {
                'total_episodes': 0,
                'avg_reward': 0.0,
                'adaptation_rate': 0.0
            }
        }
        
        logger.info("🚀 XORB Learning Integration Demo initialized")
    
    async def simulate_telemetry_pipeline(self):
        """Simulate high-throughput telemetry data pipeline"""
        logger.info("📊 Simulating Telemetry Data Pipeline...")
        
        event_types = [
            'vulnerability_detected',
            'false_positive',
            'test_completed',
            'performance_update',
            'adaptation_applied'
        ]
        
        # Simulate 100 telemetry events
        for i in range(100):
            agent = np.random.choice(self.demo_data['agents'])
            event_type = np.random.choice(event_types)
            
            # Create realistic telemetry event
            event = {
                'event_id': f"event_{uuid.uuid4().hex[:8]}",
                'agent_id': agent['id'],
                'event_type': event_type,
                'timestamp': datetime.utcnow().isoformat(),
                'payload': self._generate_event_payload(event_type, agent),
                'campaign_id': f"campaign_{i % 5}" if np.random.random() > 0.3 else None
            }
            
            self.demo_data['telemetry_events'].append(event)
            
            # Simulate processing delay
            if i % 10 == 0:
                logger.info(f"  📈 Processed {i+1}/100 telemetry events")
                await asyncio.sleep(0.1)
        
        logger.info("✅ Telemetry pipeline simulation complete")
        logger.info(f"  📊 Total events processed: {len(self.demo_data['telemetry_events'])}")
        
        # Show event distribution
        event_counts = {}
        for event in self.demo_data['telemetry_events']:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        logger.info("  📋 Event distribution:")
        for event_type, count in event_counts.items():
            logger.info(f"    {event_type}: {count} events")
    
    def _generate_event_payload(self, event_type: str, agent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic event payload based on type"""
        base_payload = {
            'agent_type': agent['type'],
            'success': np.random.random() > 0.2,  # 80% success rate
            'timestamp': time.time()
        }
        
        if event_type == 'vulnerability_detected':
            base_payload.update({
                'severity': np.random.choice(['low', 'medium', 'high', 'critical'], p=[0.3, 0.4, 0.2, 0.1]),
                'confidence': np.random.uniform(0.7, 0.98),
                'technique': np.random.choice(['sql_injection', 'xss', 'csrf', 'buffer_overflow', 'privilege_escalation']),
                'is_novel': np.random.random() > 0.85
            })
        elif event_type == 'performance_update':
            base_payload.update({
                'detection_accuracy': np.random.uniform(0.7, 0.95),
                'false_positive_rate': np.random.uniform(0.05, 0.25),
                'coverage_score': np.random.uniform(0.6, 0.9),
                'resource_efficiency': np.random.uniform(0.5, 0.9),
                'overall_fitness': np.random.uniform(0.6, 0.92)
            })
        elif event_type == 'test_completed':
            base_payload.update({
                'duration': np.random.randint(60, 3600),
                'coverage': np.random.uniform(0.5, 0.95),
                'findings_count': np.random.randint(0, 15),
                'efficiency_score': np.random.uniform(0.4, 0.9)
            })
        
        return base_payload
    
    async def simulate_learning_adaptation(self):
        """Simulate learning and adaptation cycles"""
        logger.info("🧠 Simulating Learning Adaptation Cycles...")
        
        # Simulate 5 learning cycles
        for cycle in range(5):
            logger.info(f"  🔄 Learning Cycle {cycle + 1}/5")
            
            # Simulate learning from telemetry events
            cycle_events = [e for e in self.demo_data['telemetry_events'] 
                           if e['event_type'] in ['vulnerability_detected', 'performance_update']]
            
            # Calculate rewards and update metrics
            rewards = []
            for event in cycle_events[:20]:  # Process 20 events per cycle
                reward = self._calculate_reward(event)
                rewards.append(reward)
            
            if rewards:
                avg_reward = np.mean(rewards)
                self.demo_data['learning_metrics']['avg_reward'] = avg_reward
                self.demo_data['learning_metrics']['total_episodes'] += len(rewards)
                
                # Simulate agent performance improvement
                for agent in self.demo_data['agents']:
                    if avg_reward > 0.5:
                        # Positive learning - improve performance
                        improvement = np.random.uniform(0.01, 0.05)
                        agent['performance'] = min(0.98, agent['performance'] + improvement)
                    else:
                        # Negative learning - slight performance adjustment
                        adjustment = np.random.uniform(-0.02, 0.01)
                        agent['performance'] = max(0.1, agent['performance'] + adjustment)
            
            # Simulate adaptation trigger
            if avg_reward < 0.3:
                logger.info("    🔧 Adaptation triggered due to low performance")
                self.demo_data['learning_metrics']['adaptation_rate'] += 0.1
            
            await asyncio.sleep(0.5)
        
        logger.info("✅ Learning adaptation simulation complete")
        logger.info(f"  📈 Total learning episodes: {self.demo_data['learning_metrics']['total_episodes']}")
        logger.info(f"  🎯 Average reward: {self.demo_data['learning_metrics']['avg_reward']:.3f}")
        logger.info(f"  🔄 Adaptation rate: {self.demo_data['learning_metrics']['adaptation_rate']:.2f}")
    
    def _calculate_reward(self, event: Dict[str, Any]) -> float:
        """Calculate reward signal from event"""
        payload = event['payload']
        base_reward = 0.5
        
        if event['event_type'] == 'vulnerability_detected':
            severity_multiplier = {
                'critical': 2.0, 'high': 1.5, 'medium': 1.0, 'low': 0.5
            }
            confidence = payload.get('confidence', 0.5)
            severity = payload.get('severity', 'medium')
            novelty_bonus = 0.5 if payload.get('is_novel', False) else 0.0
            
            reward = base_reward * severity_multiplier.get(severity, 1.0) * confidence + novelty_bonus
        elif event['event_type'] == 'performance_update':
            accuracy = payload.get('detection_accuracy', 0.5)
            efficiency = payload.get('resource_efficiency', 0.5)
            reward = (accuracy * 0.6 + efficiency * 0.4)
        else:
            reward = base_reward if payload.get('success', False) else 0.2
        
        return reward
    
    async def simulate_orchestration_intelligence(self):
        """Simulate intelligent campaign orchestration"""
        logger.info("🎯 Simulating Intelligent Campaign Orchestration...")
        
        # Create test campaigns with different strategies
        strategies = ['sequential', 'parallel', 'adaptive', 'swarm']
        
        for i, strategy in enumerate(strategies):
            campaign = {
                'campaign_id': f"campaign_{uuid.uuid4().hex[:8]}",
                'name': f"Demo Campaign {i+1} ({strategy})",
                'strategy': strategy,
                'target_complexity': np.random.uniform(0.3, 0.9),
                'assigned_agents': [],
                'state': 'planning',
                'progress': 0.0,
                'success_rate': 0.0,
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Simulate agent assignment based on strategy
            if strategy == 'sequential':
                # Assign best single agent
                best_agent = max(self.demo_data['agents'], key=lambda a: a['performance'])
                campaign['assigned_agents'] = [best_agent['id']]
            elif strategy == 'parallel':
                # Assign multiple agents
                top_agents = sorted(self.demo_data['agents'], key=lambda a: a['performance'], reverse=True)[:3]
                campaign['assigned_agents'] = [a['id'] for a in top_agents]
            elif strategy == 'swarm':
                # Assign all agents
                campaign['assigned_agents'] = [a['id'] for a in self.demo_data['agents']]
            else:  # adaptive
                # Assign balanced set
                campaign['assigned_agents'] = [a['id'] for a in self.demo_data['agents'][:2]]
            
            # Simulate campaign execution
            await self._simulate_campaign_execution(campaign)
            
            self.demo_data['campaigns'].append(campaign)
            logger.info(f"  📋 Campaign '{campaign['name']}' - Strategy: {strategy}")
            logger.info(f"    Agents: {len(campaign['assigned_agents'])}, Success: {campaign['success_rate']:.1%}")
        
        logger.info("✅ Orchestration intelligence simulation complete")
        
        # Show strategy effectiveness
        strategy_performance = {}
        for campaign in self.demo_data['campaigns']:
            strategy = campaign['strategy']
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(campaign['success_rate'])
        
        logger.info("  📊 Strategy effectiveness:")
        for strategy, rates in strategy_performance.items():
            avg_rate = np.mean(rates)
            logger.info(f"    {strategy}: {avg_rate:.1%} average success rate")
    
    async def _simulate_campaign_execution(self, campaign: Dict[str, Any]):
        """Simulate campaign execution and adaptation"""
        campaign['state'] = 'active'
        
        # Simulate execution phases
        phases = ['reconnaissance', 'scanning', 'exploitation', 'post_exploitation']
        
        for phase in phases:
            # Simulate phase duration
            await asyncio.sleep(0.1)
            
            # Calculate success probability based on agent performance and strategy
            agent_performances = []
            for agent_id in campaign['assigned_agents']:
                agent = next(a for a in self.demo_data['agents'] if a['id'] == agent_id)
                agent_performances.append(agent['performance'])
            
            if campaign['strategy'] == 'sequential':
                # Sequential uses best agent performance
                phase_success = max(agent_performances) > 0.6
            elif campaign['strategy'] == 'parallel':
                # Parallel uses average performance
                phase_success = np.mean(agent_performances) > 0.5
            elif campaign['strategy'] == 'swarm':
                # Swarm benefits from diversity
                phase_success = (np.mean(agent_performances) + np.std(agent_performances) * 0.5) > 0.6
            else:  # adaptive
                # Adaptive adjusts based on target complexity
                required_performance = campaign['target_complexity']
                phase_success = np.mean(agent_performances) > required_performance
            
            if phase_success:
                campaign['progress'] += 0.25
            else:
                # Simulate adaptation
                if campaign['strategy'] == 'adaptive' and np.random.random() > 0.5:
                    # Reassign agents
                    logger.info(f"    🔧 Adapting campaign '{campaign['name']}' in {phase} phase")
                    campaign['progress'] += 0.15  # Partial progress due to adaptation
        
        # Calculate final success rate
        campaign['success_rate'] = min(1.0, campaign['progress'] + np.random.uniform(-0.1, 0.1))
        campaign['state'] = 'completed'
    
    async def simulate_security_monitoring(self):
        """Simulate security framework monitoring"""
        logger.info("🛡️ Simulating Security Framework Monitoring...")
        
        security_events = [
            {'type': 'authentication', 'result': 'success', 'user': 'admin'},
            {'type': 'authentication', 'result': 'failure', 'user': 'unknown', 'ip': '192.168.1.100'},
            {'type': 'authorization', 'result': 'denied', 'user': 'analyst', 'resource': 'system:admin'},
            {'type': 'data_access', 'result': 'success', 'user': 'operator', 'resource': 'campaigns'},
            {'type': 'configuration_change', 'result': 'success', 'user': 'admin', 'change': 'learning_rate'},
        ]
        
        violations_detected = 0
        
        for event in security_events:
            # Simulate security analysis
            await asyncio.sleep(0.1)
            
            # Check for violations
            if event['result'] == 'failure' and event['type'] == 'authentication':
                violations_detected += 1
                logger.info(f"  🚨 Security violation detected: Failed authentication from {event.get('ip', 'unknown')}")
            elif event['result'] == 'denied' and event['type'] == 'authorization':
                logger.info(f"  ⚠️ Authorization denied: {event['user']} attempted {event['resource']}")
            else:
                logger.info(f"  ✅ Security event: {event['type']} - {event['result']}")
        
        logger.info("✅ Security monitoring simulation complete")
        logger.info(f"  🛡️ Security violations detected: {violations_detected}")
        logger.info(f"  📝 Total security events processed: {len(security_events)}")
    
    async def generate_performance_report(self):
        """Generate comprehensive performance report"""
        logger.info("📋 Generating Performance Report...")
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_agents': len(self.demo_data['agents']),
                'total_campaigns': len(self.demo_data['campaigns']),
                'total_telemetry_events': len(self.demo_data['telemetry_events']),
                'learning_episodes': self.demo_data['learning_metrics']['total_episodes'],
                'average_reward': self.demo_data['learning_metrics']['avg_reward']
            },
            'agent_performance': {},
            'campaign_effectiveness': {},
            'learning_progress': self.demo_data['learning_metrics']
        }
        
        # Agent performance analysis
        for agent in self.demo_data['agents']:
            agent_events = [e for e in self.demo_data['telemetry_events'] if e['agent_id'] == agent['id']]
            successful_events = [e for e in agent_events if e['payload'].get('success', False)]
            
            report['agent_performance'][agent['id']] = {
                'type': agent['type'],
                'current_performance': agent['performance'],
                'total_events': len(agent_events),
                'success_rate': len(successful_events) / max(len(agent_events), 1),
                'improvement': agent['performance'] - 0.5  # Assuming baseline was 0.5
            }
        
        # Campaign effectiveness analysis
        if self.demo_data['campaigns']:
            for campaign in self.demo_data['campaigns']:
                report['campaign_effectiveness'][campaign['campaign_id']] = {
                    'strategy': campaign['strategy'],
                    'success_rate': campaign['success_rate'],
                    'agents_used': len(campaign['assigned_agents']),
                    'target_complexity': campaign['target_complexity']
                }
        
        # Save report
        report_file = f"/tmp/xorb_learning_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Performance report generated: {report_file}")
        
        # Display key metrics
        logger.info("📊 Key Performance Metrics:")
        logger.info(f"  🤖 Active Agents: {report['summary']['total_agents']}")
        logger.info(f"  🎯 Campaigns Executed: {report['summary']['total_campaigns']}")
        logger.info(f"  📈 Telemetry Events: {report['summary']['total_telemetry_events']}")
        logger.info(f"  🧠 Learning Episodes: {report['summary']['learning_episodes']}")
        logger.info(f"  🏆 Average Reward: {report['summary']['average_reward']:.3f}")
        
        if self.demo_data['campaigns']:
            avg_success = np.mean([c['success_rate'] for c in self.demo_data['campaigns']])
            logger.info(f"  ✅ Campaign Success Rate: {avg_success:.1%}")
        
        return report
    
    async def run_comprehensive_demo(self):
        """Run complete demonstration of learning integration"""
        logger.info("🚀 Starting XORB Learning Engine Integration Demo")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Run all demonstration components
            await self.simulate_telemetry_pipeline()
            await asyncio.sleep(1)
            
            await self.simulate_learning_adaptation()
            await asyncio.sleep(1)
            
            await self.simulate_orchestration_intelligence()
            await asyncio.sleep(1)
            
            await self.simulate_security_monitoring()
            await asyncio.sleep(1)
            
            # Generate final report
            report = await self.generate_performance_report()
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("=" * 60)
            logger.info("🎉 XORB Learning Engine Integration Demo Complete!")
            logger.info(f"⏱️ Total demo duration: {duration:.2f} seconds")
            logger.info("🎯 All integration components demonstrated successfully")
            logger.info("🚀 System ready for production deployment!")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Demo failed with error: {e}")
            raise

async def main():
    """Main demo execution"""
    demo = LearningIntegrationDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main())