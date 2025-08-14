import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { ComplianceStandard } from '../entities/compliance-standard.entity';
import { ComplianceFinding } from '../entities/compliance-finding.entity';
import { ComplianceReport } from '../entities/compliance-report.entity';
import { XORBComplianceAPI } from '../integrations/xorb-compliance.api';

@Injectable()
export class ComplianceDeltaTrackingService {
  constructor(
    @InjectRepository(ComplianceReport)
    private readonly complianceReportRepository: Repository<ComplianceReport>,
    @InjectRepository(ComplianceFinding)
    private readonly complianceFindingRepository: Repository<ComplianceFinding>,
    private readonly xorbComplianceAPI: XORBComplianceAPI
  ) {}

  /**
   * Get daily compliance changes for a specific standard and time period
   * @param standardId ID of the compliance standard
   * @param startDate Start date for the tracking period
   * @param endDate End date for the tracking period
   * @returns ComplianceDeltaTrackingResult containing daily changes
   */
  async getDailyComplianceChanges(
    standardId: string,
    startDate: Date,
    endDate: Date
  ): Promise<ComplianceDeltaTrackingResult> {
    // Get initial compliance state at start date
    const initialReport = await this.complianceReportRepository.findOne({
      where: {
        standard: { id: standardId },
        createdAt: LessThanOrEqual(startDate)
      },
      order: { createdAt: 'DESC' },
      relations: ['findings']
    });

    // Get final compliance state at end date
    const finalReport = await this.complianceReportRepository.findOne({
      where: {
        standard: { id: standardId },
        createdAt: LessThanOrEqual(endDate)
      },
      order: { createdAt: 'DESC' },
      relations: ['findings']
    });

    if (!initialReport || !finalReport) {
      throw new Error('No compliance reports found for the specified period');
    }

    // Calculate daily changes between dates
    const dailyChanges = await this.calculateDailyChanges(
      standardId,
      startDate,
      endDate
    );

    // Get findings that were resolved during the period
    const resolvedFindings = this.getResolvedFindings(
      initialReport.findings,
      finalReport.findings
    );

    // Get findings that were newly identified during the period
    const newFindings = this.getNewFindings(
      initialReport.findings,
      finalReport.findings
    );

    // Get findings that had severity changes during the period
    const severityChangedFindings = this.getSeverityChangedFindings(
      initialReport.findings,
      finalReport.findings
    );

    return {
      standardId,
      startDate,
      endDate,
      initialComplianceScore: initialReport.complianceScore,
      finalComplianceScore: finalReport.complianceScore,
      totalFindingsResolved: resolvedFindings.length,
      totalNewFindings: newFindings.length,
      totalSeverityChanges: severityChangedFindings.length,
      dailyComplianceTrend: dailyChanges,
      resolvedFindings,
      newFindings,
      severityChangedFindings
    };
  }

  /**
   * Calculate daily compliance changes between two dates
   * @param standardId ID of the compliance standard
   * @param startDate Start date
   * @param endDate End date
   * @returns Array of daily compliance data points
   */
  private async calculateDailyChanges(
    standardId: string,
    startDate: Date,
    endDate: Date
  ): Promise<DailyComplianceDataPoint[]> {
    const dailyData: DailyComplianceDataPoint[] = [];
    const currentDate = new Date(startDate);

    while (currentDate <= endDate) {
      // Get the latest report for this day
      const report = await this.complianceReportRepository.findOne({
        where: {
          standard: { id: standardId },
          createdAt: LessThanOrEqual(currentDate)
        },
        order: { createdAt: 'DESC' },
        relations: ['findings']
      });

      if (report) {
        dailyData.push({
          date: new Date(currentDate),
          complianceScore: report.complianceScore,
          totalFindings: report.findings.length,
          criticalFindings: report.findings.filter(f => f.severity === 'CRITICAL').length,
          highFindings: report.findings.filter(f => f.severity === 'HIGH').length,
          mediumFindings: report.findings.filter(f => f.severity === 'MEDIUM').length,
          lowFindings: report.findings.filter(f => f.severity === 'LOW').length
        });
      }

      // Move to next day
      currentDate.setDate(currentDate.getDate() + 1);
    }

    return dailyData;
  }

  /**
   * Get findings that were resolved between two reports
   * @param initialFindings Findings from the initial report
   * @param finalFindings Findings from the final report
   * @returns Array of resolved findings
   */
  private getResolvedFindings(
    initialFindings: ComplianceFinding[],
    finalFindings: ComplianceFinding[]
  ): ComplianceFinding[] {
    return initialFindings.filter(initialFinding =>
      !finalFindings.some(finalFinding => finalFinding.id === initialFinding.id)
    );
  }

  /**
   * Get findings that were newly identified between two reports
   * @param initialFindings Findings from the initial report
   * @param finalFindings Findings from the final report
   * @returns Array of new findings
   */
  private getNewFindings(
    initialFindings: ComplianceFinding[],
    finalFindings: ComplianceFinding[]
  ): ComplianceFinding[] {
    return finalFindings.filter(finalFinding =>
      !initialFindings.some(initialFinding => initialFinding.id === finalFinding.id)
    );
  }

  /**
   * Get findings that had severity changes between two reports
   * @param initialFindings Findings from the initial report
   * @param finalFindings Findings from the final report
   * @returns Array of findings with severity changes
   */
  private getSeverityChangedFindings(
    initialFindings: ComplianceFinding[],
    finalFindings: ComplianceFinding[]
  ): FindingSeverityChange[] {
    return finalFindings
      .map(finalFinding => {
        const initialFinding = initialFindings.find(
          f => f.id === finalFinding.id
        );

        if (initialFinding && initialFinding.severity !== finalFinding.severity) {
          return {
            finding: finalFinding,
            previousSeverity: initialFinding.severity,
            newSeverity: finalFinding.severity
          };
        }

        return null;
      })
      .filter(change => change !== null) as FindingSeverityChange[];
  }

  /**
   * Generate a compliance delta summary for a specific standard
   * @param standardId ID of the compliance standard
   * @param startDate Start date for the tracking period
   * @param endDate End date for the tracking period
   * @returns ComplianceDeltaSummary containing key metrics
   */
  async generateDeltaSummary(
    standardId: string,
    startDate: Date,
    endDate: Date
  ): Promise<ComplianceDeltaSummary> {
    const deltaResult = await this.getDailyComplianceChanges(
      standardId,
      startDate,
      endDate
    );

    // Calculate overall compliance change
    const complianceChange =
      deltaResult.finalComplianceScore - deltaResult.initialComplianceScore;

    // Calculate net severity changes
    const netSeverityChanges = this.calculateNetSeverityChanges(
      deltaResult.severityChangedFindings
    );

    return {
      standardId,
      startDate,
      endDate,
      initialComplianceScore: deltaResult.initialComplianceScore,
      finalComplianceScore: deltaResult.finalComplianceScore,
      complianceChange,
      complianceTrend: complianceChange > 0 ? 'IMPROVED' :
                        (complianceChange < 0 ? 'DECLINED' : 'STABLE'),
      findingsResolved: deltaResult.totalFindingsResolved,
      findingsAdded: deltaResult.totalNewFindings,
      severityImprovements: netSeverityChanges.improvements,
      severityDegradations: netSeverityChanges.degradations,
      netSeverityChange: netSeverityChanges.netChange,
      keyContributors: this.identifyKeyContributors(deltaResult)
    };
  }

  /**
   * Calculate net severity changes from finding severity transitions
   * @param severityChanges Array of finding severity changes
   * @returns Net severity change metrics
   */
  private calculateNetSeverityChanges(
    severityChanges: FindingSeverityChange[]
  ): NetSeverityChangeMetrics {
    let improvements = 0;
    let degradations = 0;

    severityChanges.forEach(change => {
      const severityOrder = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
      const previousIndex = severityOrder.indexOf(change.previousSeverity);
      const newIndex = severityOrder.indexOf(change.newSeverity);

      if (newIndex < previousIndex) {
        improvements++;
      } else if (newIndex > previousIndex) {
        degradations++;
      }
    });

    return {
      improvements,
      degradations,
      netChange: improvements - degradations
    };
  }

  /**
   * Identify key contributors to compliance changes
   * @param deltaResult Compliance delta tracking result
   * @returns Array of key contributors
   */
  private identifyKeyContributors(
    deltaResult: ComplianceDeltaTrackingResult
  ): ComplianceKeyContributor[] {
    const keyContributors: ComplianceKeyContributor[] = [];

    // Add top resolved findings by severity
    const topResolvedBySeverity = [...deltaResult.resolvedFindings]
      .sort((a, b) => this.getSeverityValue(b.severity) - this.getSeverityValue(a.severity))
      .slice(0, 5);

    if (topResolvedBySeverity.length > 0) {
      keyContributors.push({
        type: 'FINDINGS_RESOLVED',
        description: `Resolved top ${topResolvedBySeverity.length} findings by severity`,
        findings: topResolvedBySeverity
      });
    }

    // Add top new findings by severity
    const topNewBySeverity = [...deltaResult.newFindings]
      .sort((a, b) => this.getSeverityValue(b.severity) - this.getSeverityValue(a.severity))
      .slice(0, 5);

    if (topNewBySeverity.length > 0) {
      keyContributors.push({
        type: 'FINDINGS_ADDED',
        description: `New findings with highest severity`,
        findings: topNewBySeverity
      });
    }

    // Add significant severity improvements
    const significantImprovements = deltaResult.severityChangedFindings
      .filter(change => this.getSeverityValue(change.previousSeverity) > this.getSeverityValue(change.newSeverity))
      .sort((a, b) =>
        (this.getSeverityValue(a.previousSeverity) - this.getSeverityValue(a.newSeverity)) -
        (this.getSeverityValue(b.previousSeverity) - this.getSeverityValue(b.newSeverity))
      )
      .slice(0, 3);

    if (significantImprovements.length > 0) {
      keyContributors.push({
        type: 'SEVERITY_IMPROVED',
        description: `Most significant severity improvements`,
        severityChanges: significantImprovements
      });
    }

    // Add significant severity degradations
    const significantDegradations = deltaResult.severityChangedFindings
      .filter(change => this.getSeverityValue(change.previousSeverity) < this.getSeverityValue(change.newSeverity))
      .sort((a, b) =>
        (this.getSeverityValue(b.newSeverity) - this.getSeverityValue(b.previousSeverity)) -
        (this.getSeverityValue(a.newSeverity) - this.getSeverityValue(a.previousSeverity))
      )
      .slice(0, 3);

    if (significantDegradations.length > 0) {
      keyContributors.push({
        type: 'SEVERITY_DEGRADED',
        description: `Most significant severity degradations`,
        severityChanges: significantDegradations
      });
    }

    return keyContributors;
  }

  /**
   * Get numeric value for severity level
   * @param severity Severity level
   * @returns Numeric value (higher is more severe)
   */
  private getSeverityValue(severity: string): number {
    switch (severity.toUpperCase()) {
      case 'CRITICAL': return 4;
      case 'HIGH': return 3;
      case 'MEDIUM': return 2;
      case 'LOW': return 1;
      default: return 0;
    }
  }
}

// Types and interfaces
export interface ComplianceDeltaTrackingResult {
  standardId: string;
  startDate: Date;
  endDate: Date;
  initialComplianceScore: number;
  finalComplianceScore: number;
  totalFindingsResolved: number;
  totalNewFindings: number;
  totalSeverityChanges: number;
  dailyComplianceTrend: DailyComplianceDataPoint[];
  resolvedFindings: ComplianceFinding[];
  newFindings: ComplianceFinding[];
  severityChangedFindings: FindingSeverityChange[];
}

export interface DailyComplianceDataPoint {
  date: Date;
  complianceScore: number;
  totalFindings: number;
  criticalFindings: number;
  highFindings: number;
  mediumFindings: number;
  lowFindings: number;
}

export interface FindingSeverityChange {
  finding: ComplianceFinding;
  previousSeverity: string;
  newSeverity: string;
}

export interface ComplianceDeltaSummary {
  standardId: string;
  startDate: Date;
  endDate: Date;
  initialComplianceScore: number;
  finalComplianceScore: number;
  complianceChange: number;
  complianceTrend: 'IMPROVED' | 'DECLINED' | 'STABLE';
  findingsResolved: number;
  findingsAdded: number;
  severityImprovements: number;
  severityDegradations: number;
  netSeverityChange: number;
  keyContributors: ComplianceKeyContributor[];
}

export interface NetSeverityChangeMetrics {
  improvements: number;
  degradations: number;
  netChange: number;
}

export interface ComplianceKeyContributor {
  type: 'FINDINGS_RESOLVED' | 'FINDINGS_ADDED' | 'SEVERITY_IMPROVED' | 'SEVERITY_DEGRADED';
  description: string;
  findings?: ComplianceFinding[];
  severityChanges?: FindingSeverityChange[];
}
