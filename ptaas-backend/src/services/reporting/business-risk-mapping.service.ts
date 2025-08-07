import { Injectable } from '@nestjs/common';
import { Vulnerability } from '../vulnerability/vulnerability.entity';
import { BusinessRiskAssessment } from './dto/business-risk-assessment.dto';
import { RiskImpactLevel } from './enums/risk-impact-level.enum';
import { RiskCategory } from './enums/risk-category.enum';

@Injectable()
export class BusinessRiskMappingService {
  private readonly financialImpactFactors = {
    [RiskCategory.CRITICAL_SYSTEM]: 0.8,
    [RiskCategory.DATA_BREACH]: 0.95,
    [RiskCategory.COMPLIANCE]: 0.7,
    [RiskCategory.OPERATIONAL]: 0.6,
    [RiskCategory.REPUTATIONAL]: 0.85,
  };

  private readonly operationalImpactFactors = {
    [RiskCategory.CRITICAL_SYSTEM]: 0.9,
    [RiskCategory.DATA_BREACH]: 0.7,
    [RiskCategory.COMPLIANCE]: 0.65,
    [RiskCategory.OPERATIONAL]: 0.8,
    [RiskCategory.REPUTATIONAL]: 0.5,
  };

  calculateBusinessRisk(vulnerability: Vulnerability): BusinessRiskAssessment {
    const riskCategory = this.determineRiskCategory(vulnerability);
    
    const financialImpact = this.calculateFinancialImpact(vulnerability, riskCategory);
    const operationalImpact = this.calculateOperationalImpact(vulnerability, riskCategory);
    const overallRiskScore = this.calculateOverallRiskScore(financialImpact, operationalImpact, vulnerability.severity);
    
    return {
      vulnerabilityId: vulnerability.id,
      riskCategory,
      financialImpact,
      operationalImpact,
      overallRiskScore,
      riskLevel: this.determineRiskLevel(overallRiskScore),
      recommendations: this.generateRecommendations(riskCategory, vulnerability.severity)
    };
  }

  private determineRiskCategory(vulnerability: Vulnerability): RiskCategory {
    // Implementation logic based on vulnerability type, affected assets, and threat intel
    // This is a simplified example - in production, this would use more sophisticated logic
    if (vulnerability.type.includes('authentication') || vulnerability.type.includes('access')) {
      return RiskCategory.COMPLIANCE;
    } else if (vulnerability.type.includes('data') || vulnerability.type.includes('exposure')) {
      return RiskCategory.DATA_BREACH;
    } else if (vulnerability.type.includes('denial') || vulnerability.type.includes('availability')) {
      return RiskCategory.OPERATIONAL;
    } else if (vulnerability.type.includes('critical')) {
      return RiskCategory.CRITICAL_SYSTEM;
    } else {
      return RiskCategory.REPUTATIONAL;
    }
  }

  private calculateFinancialImpact(vulnerability: Vulnerability, riskCategory: RiskCategory): number {
    // Base calculation on CVSS score, vulnerability type, and asset value
    const baseImpact = vulnerability.severity * this.financialImpactFactors[riskCategory];
    
    // Adjust based on asset value
    const assetValueMultiplier = this.getAssetValueMultiplier(vulnerability.affectedAssetValue);
    
    // Adjust based on potential regulatory fines
    const regulatoryMultiplier = this.getRegulatoryMultiplier(riskCategory);
    
    return Math.min(1.0, baseImpact * assetValueMultiplier * regulatoryMultiplier);
  }

  private calculateOperationalImpact(vulnerability: Vulnerability, riskCategory: RiskCategory): number {
    // Base calculation on CVSS score, vulnerability type, and asset criticality
    const baseImpact = vulnerability.severity * this.operationalImpactFactors[riskCategory];
    
    // Adjust based on asset criticality
    const criticalityMultiplier = this.getAssetCriticalityMultiplier(vulnerability.affectedAssetCriticality);
    
    // Adjust based on potential service disruption
    const disruptionMultiplier = this.getDisruptionMultiplier(vulnerability.type);
    
    return Math.min(1.0, baseImpact * criticalityMultiplier * disruptionMultiplier);
  }

  private calculateOverallRiskScore(financialImpact: number, operationalImpact: number, severity: number): number {
    // Weighted average based on severity and impact factors
    const financialWeight = 0.4;
    const operationalWeight = 0.4;
    const severityWeight = 0.2;
    
    return (
      financialImpact * financialWeight +
      operationalImpact * operationalWeight +
      severity * severityWeight
    );
  }

  private determineRiskLevel(riskScore: number): string {
    if (riskScore >= 0.8) return 'critical';
    if (riskScore >= 0.6) return 'high';
    if (riskScore >= 0.4) return 'medium';
    return 'low';
  }

  private generateRecommendations(riskCategory: RiskCategory, severity: number): string[] {
    const recommendations: string[] = [];
    
    // Common recommendations
    recommendations.push('Apply vendor patches or implement mitigations');
    recommendations.push('Implement network segmentation and access controls');
    
    // Category-specific recommendations
    switch (riskCategory) {
      case RiskCategory.CRITICAL_SYSTEM:
        recommendations.push('Implement redundant systems and failover mechanisms');
        recommendations.push('Conduct regular penetration testing on critical systems');
        break;
      case RiskCategory.DATA_BREACH:
        recommendations.push('Implement data encryption at rest and in transit');
        recommendations.push('Review and strengthen access control policies');
        break;
      case RiskCategory.COMPLIANCE:
        recommendations.push('Review compliance requirements and implement necessary controls');
        recommendations.push('Conduct regular compliance audits');
        break;
      case RiskCategory.OPERATIONAL:
        recommendations.push('Implement monitoring and alerting for system availability');
        recommendations.push('Develop and test incident response plans');
        break;
      case RiskCategory.REPUTATIONAL:
        recommendations.push('Implement web application firewalls to prevent attacks');
        recommendations.push('Monitor for brand abuse and reputation risks');
        break;
    }
    
    // Severity-based recommendations
    if (severity >= 8.0) {
      recommendations.push('Implement immediate mitigation measures');
      recommendations.push('Conduct a thorough security review of affected systems');
    } else if (severity >= 6.0) {
      recommendations.push('Implement mitigation measures within 30 days');
      recommendations.push('Monitor for exploitation attempts');
    }
    
    return recommendations;
  }

  private getAssetValueMultiplier(assetValue: string): number {
    switch (assetValue.toLowerCase()) {
      case 'high': return 1.3;
      case 'medium': return 1.0;
      case 'low': return 0.7;
      default: return 1.0;
    }
  }

  private getAssetCriticalityMultiplier(assetCriticality: string): number {
    switch (assetCriticality.toLowerCase()) {
      case 'high': return 1.5;
      case 'medium': return 1.0;
      case 'low': return 0.6;
      default: return 1.0;
    }
  }

  private getRegulatoryMultiplier(riskCategory: RiskCategory): number {
    if (riskCategory === RiskCategory.COMPLIANCE || riskCategory === RiskCategory.DATA_BREACH) {
      return 1.4; // Higher multiplier for compliance-related risks
    }
    return 1.0;
  }

  private getDisruptionMultiplier(vulnerabilityType: string): number {
    if (vulnerabilityType.includes('denial') || vulnerabilityType.includes('availability')) {
      return 1.6; // Higher multiplier for availability risks
    }
    return 1.0;
  }
}