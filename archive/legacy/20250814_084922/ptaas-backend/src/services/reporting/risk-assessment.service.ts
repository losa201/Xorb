import { Injectable } from '@nestjs/common';
import { Vulnerability } from '../vulnerability.model';

interface FinancialImpactFactors {
  directCosts: number; // Direct financial losses in EUR
  indirectCosts: number; // Indirect financial losses in EUR
  regulatoryFines: number; // Potential regulatory fines in EUR
  remediationCosts: number; // Estimated remediation costs in EUR
}

interface OperationalImpactFactors {
  systemDowntime: number; // Potential downtime in hours
  dataLoss: number; // Potential data loss in GB
  serviceDisruption: number; // Service disruption level (1-5)
  reputationImpact: number; // Reputation impact level (1-5)
}

interface RiskAssessment {
  id: string;
  vulnerabilityId: string;
  financialImpact: FinancialImpactFactors;
  operationalImpact: OperationalImpactFactors;
  riskScore: number; // Overall risk score (1-10)
  riskLevel: 'Low' | 'Medium' | 'High' | 'Critical';
  createdAt: Date;
  updatedAt: Date;
}

@Injectable()
export class RiskAssessmentService {
  private riskAssessments: RiskAssessment[] = [];

  // Create a new risk assessment for a vulnerability
  createAssessment(vulnerability: Vulnerability): RiskAssessment {
    const riskAssessment = this.calculateRiskAssessment(vulnerability);
    this.riskAssessments.push(riskAssessment);
    return riskAssessment;
  }

  // Get risk assessment by vulnerability ID
  getAssessmentByVulnerabilityId(vulnerabilityId: string): RiskAssessment | undefined {
    return this.riskAssessments.find(assessment => assessment.vulnerabilityId === vulnerabilityId);
  }

  // Get all risk assessments
  getAllAssessments(): RiskAssessment[] {
    return this.riskAssessments;
  }

  // Update a risk assessment
  updateAssessment(vulnerabilityId: string, updates: Partial<FinancialImpactFactors & OperationalImpactFactors>): RiskAssessment | undefined {
    const assessment = this.riskAssessments.find(assessment => assessment.vulnerabilityId === vulnerabilityId);

    if (assessment) {
      // Update financial impact factors if provided
      if (updates.directCosts !== undefined) assessment.financialImpact.directCosts = updates.directCosts;
      if (updates.indirectCosts !== undefined) assessment.financialImpact.indirectCosts = updates.indirectCosts;
      if (updates.regulatoryFines !== undefined) assessment.financialImpact.regulatoryFines = updates.regulatoryFines;
      if (updates.remediationCosts !== undefined) assessment.financialImpact.remediationCosts = updates.remediationCosts;

      // Update operational impact factors if provided
      if (updates.systemDowntime !== undefined) assessment.operationalImpact.systemDowntime = updates.systemDowntime;
      if (updates.dataLoss !== undefined) assessment.operationalImpact.dataLoss = updates.dataLoss;
      if (updates.serviceDisruption !== undefined) assessment.operationalImpact.serviceDisruption = updates.serviceDisruption;
      if (updates.reputationImpact !== undefined) assessment.operationalImpact.reputationImpact = updates.reputationImpact;

      // Recalculate risk score and level
      assessment.riskScore = this.calculateRiskScore(assessment);
      assessment.riskLevel = this.calculateRiskLevel(assessment.riskScore);
      assessment.updatedAt = new Date();

      return assessment;
    }

    return undefined;
  }

  // Calculate initial risk assessment based on vulnerability data
  private calculateRiskAssessment(vulnerability: Vulnerability): RiskAssessment {
    // Base factors from vulnerability data
    const baseFactors = {
      directCosts: this.calculateDirectCosts(vulnerability),
      indirectCosts: this.calculateIndirectCosts(vulnerability),
      regulatoryFines: this.calculateRegulatoryFines(vulnerability),
      remediationCosts: this.calculateRemediationCosts(vulnerability)
    };

    const operationalFactors = {
      systemDowntime: this.calculateSystemDowntime(vulnerability),
      dataLoss: this.calculateDataLoss(vulnerability),
      serviceDisruption: this.calculateServiceDisruption(vulnerability),
      reputationImpact: this.calculateReputationImpact(vulnerability)
    };

    const riskScore = this.calculateRiskScore({
      financialImpact: baseFactors,
      operationalImpact: operationalFactors
    });

    return {
      id: `RA-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      vulnerabilityId: vulnerability.id,
      financialImpact: baseFactors,
      operationalImpact: operationalFactors,
      riskScore,
      riskLevel: this.calculateRiskLevel(riskScore),
      createdAt: new Date(),
      updatedAt: new Date()
    };
  }

  // Calculate risk score based on financial and operational impacts
  private calculateRiskScore(assessment: Omit<RiskAssessment, 'id' | 'vulnerabilityId' | 'riskLevel' | 'createdAt' | 'updatedAt'>): number {
    // Calculate financial impact score (0-100)
    const financialImpactScore = (
      assessment.financialImpact.directCosts * 0.3 +
      assessment.financialImpact.indirectCosts * 0.2 +
      assessment.financialImpact.regulatoryFines * 0.3 +
      assessment.financialImpact.remediationCosts * 0.2
    ) / 1000; // Normalize to 0-100 scale

    // Calculate operational impact score (0-100)
    const operationalImpactScore = (
      assessment.operationalImpact.systemDowntime * 0.25 +
      assessment.operationalImpact.dataLoss * 0.25 +
      assessment.operationalImpact.serviceDisruption * 10 * 0.25 +
      assessment.operationalImpact.reputationImpact * 10 * 0.25
    ); // Already on a 0-100 scale

    // Combine financial and operational impacts (50/50 weighting)
    const riskScore = (financialImpactScore * 0.5) + (operationalImpactScore * 0.5);

    // Cap the risk score at 10
    return Math.min(riskScore / 10, 10);
  }

  // Calculate risk level based on risk score
  private calculateRiskLevel(riskScore: number): 'Low' | 'Medium' | 'High' | 'Critical' {
    if (riskScore < 3) return 'Low';
    if (riskScore < 6) return 'Medium';
    if (riskScore < 8) return 'High';
    return 'Critical';
  }

  // Calculate direct costs based on vulnerability type and severity
  private calculateDirectCosts(vulnerability: Vulnerability): number {
    // Implementation details would consider factors like:
    // - Vulnerability severity (CVSS score)
    // - Type of vulnerability
    // - Affected systems
    // - Industry-specific factors
    const baseCost = vulnerability.cvssScore * 1000;
    return baseCost;
  }

  // Calculate indirect costs based on business context
  private calculateIndirectCosts(vulnerability: Vulnerability): number {
    // Implementation details would consider factors like:
    // - Business criticality of affected systems
    // - Potential customer impact
    // - Industry-specific factors
    const baseCost = vulnerability.cvssScore * 500;
    return baseCost;
  }

  // Calculate potential regulatory fines
  private calculateRegulatoryFines(vulnerability: Vulnerability): number {
    // Implementation details would consider factors like:
    // - Data protection regulations applicable (GDPR, CCPA, etc.)
    // - Type of data potentially exposed
    // - Industry-specific regulations
    const baseFine = vulnerability.cvssScore * 2000;
    return baseFine;
  }

  // Calculate estimated remediation costs
  private calculateRemediationCosts(vulnerability: Vulnerability): number {
    // Implementation details would consider factors like:
    // - Complexity of fix
    // - Number of affected systems
    // - Required resources
    const baseCost = vulnerability.cvssScore * 800;
    return baseCost;
  }

  // Estimate potential system downtime
  private calculateSystemDowntime(vulnerability: Vulnerability): number {
    // Implementation details would consider factors like:
    // - Criticality of affected systems
    // - Complexity of vulnerability
    // - Potential for exploitation
    const baseDowntime = vulnerability.cvssScore * 2;
    return baseDowntime;
  }

  // Estimate potential data loss
  private calculateDataLoss(vulnerability: Vulnerability): number {
    // Implementation details would consider factors like:
    // - Type of data exposed
    // - Number of records potentially affected
    // - System architecture
    const baseDataLoss = vulnerability.cvssScore * 5;
    return baseDataLoss;
  }

  // Estimate service disruption level
  private calculateServiceDisruption(vulnerability: Vulnerability): number {
    // Implementation details would consider factors like:
    // - Criticality of affected services
    // - Potential impact on business operations
    // - Recovery time objective (RTO)
    const baseDisruption = Math.min(vulnerability.cvssScore, 5);
    return baseDisruption;
  }

  // Estimate reputation impact level
  private calculateReputationImpact(vulnerability: Vulnerability): number {
    // Implementation details would consider factors like:
    // - Public visibility of vulnerability
    // - Industry reputation requirements
    // - Potential media attention
    const baseImpact = Math.min(vulnerability.cvssScore, 5);
    return baseImpact;
  }
}
