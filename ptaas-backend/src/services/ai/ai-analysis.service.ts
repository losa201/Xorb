import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';
import { Vulnerability } from '../vulnerabilities/vulnerability.entity';
import { VulnerabilityRepository } from '../vulnerabilities/vulnerability.repository';
import { RemediationSuggestion } from './dto/remediation-suggestion.dto';
import { AttackChain } from './dto/attack-chain.dto';
import { RiskPriorityScore } from './dto/risk-priority-score.dto';
import { FalsePositiveDetection } from './dto/false-positive-detection.dto';

@Injectable()
export class AiAnalysisService {
  private readonly xorbApiUrl: string;
  private readonly xorbApiKey: string;
  private readonly xorbAiModel: string;

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService,
    private readonly vulnerabilityRepository: VulnerabilityRepository
  ) {
    this.xorbApiUrl = this.configService.get<string>('XORB_AI_API_URL');
    this.xorbApiKey = this.configService.get<string>('XORB_AI_API_KEY');
    this.xorbAiModel = this.configService.get<string>('XORB_AI_MODEL');
  }

  /**
   * Analyze vulnerabilities with XORB AI to generate risk priority scores
   * @param vulnerabilityIds Array of vulnerability IDs to analyze
   * @param tenantId ID of the tenant for context
   * @returns Array of risk priority scores
   */
  async generateRiskPriorityScores(vulnerabilityIds: string[], tenantId: string): Promise<RiskPriorityScore[]> {
    const vulnerabilities = await this.vulnerabilityRepository.findByIds(vulnerabilityIds);
    
    const payload = {
      model: this.xorbAiModel,
      input: {
        vulnerabilities: vulnerabilities.map(vuln => ({
          id: vuln.id,
          title: vuln.title,
          description: vuln.description,
          cvssScore: vuln.cvssScore,
          assetContext: vuln.assetContext,
          publicExposure: vuln.publicExposure
        })),
        tenantContext: await this.getTenantContext(tenantId)
      }
    };

    try {
      const { data } = await firstValueFrom(
        this.httpService.post(`${this.xorbApiUrl}/risk-prioritization`, payload, {
          headers: { 
            'Authorization': `Bearer ${this.xorbApiKey}`,
            'Content-Type': 'application/json'
          }
        })
      );

      // Process and validate AI response
      return this.processRiskPriorityScoresResponse(data);
    } catch (error) {
      this.handleError(error, 'Risk prioritization analysis');
      return [];
    }
  }

  /**
   * Generate remediation guidance for vulnerabilities in specified language
   * @param vulnerabilityIds Array of vulnerability IDs
   * @param language Language code (en/de)
   * @returns Remediation suggestions
   */
  async generateRemediationGuidance(vulnerabilityIds: string[], language: string): Promise<RemediationSuggestion[]> {
    const vulnerabilities = await this.vulnerabilityRepository.findByIds(vulnerabilityIds);
    
    const payload = {
      model: this.xorbAiModel,
      input: {
        vulnerabilities: vulnerabilities.map(vuln => ({
          id: vuln.id,
          title: vuln.title,
          description: vuln.description,
          technicalDetails: vuln.technicalDetails,
          cvssScore: vuln.cvssScore
        })),
        language: language
      }
    };

    try {
      const { data } = await firstValueFrom(
        this.httpService.post(`${this.xorbApiUrl}/remediation-guidance`, payload, {
          headers: { 
            'Authorization': `Bearer ${this.xorbApiKey}`,
            'Content-Type': 'application/json'
          }
        })
      );

      // Process and validate AI response
      return this.processRemediationGuidanceResponse(data);
    } catch (error) {
      this.handleError(error, 'Remediation guidance generation');
      return [];
    }
  }

  /**
   * Detect attack chains from vulnerabilities
   * @param vulnerabilityIds Array of vulnerability IDs
   * @param tenantId ID of the tenant for context
   * @returns Attack chains
   */
  async detectAttackChains(vulnerabilityIds: string[], tenantId: string): Promise<AttackChain[]> {
    const vulnerabilities = await this.vulnerabilityRepository.findByIds(vulnerabilityIds);
    
    const payload = {
      model: this.xorbAiModel,
      input: {
        vulnerabilities: vulnerabilities.map(vuln => ({
          id: vuln.id,
          title: vuln.title,
          description: vuln.description,
          cvssScore: vuln.cvssScore,
          assetContext: vuln.assetContext,
          networkLocation: vuln.networkLocation
        })),
        tenantContext: await this.getTenantContext(tenantId)
      }
    };

    try {
      const { data } = await firstValueFrom(
        this.httpService.post(`${this.xorbApiUrl}/attack-chain-detection`, payload, {
          headers: { 
            'Authorization': `Bearer ${this.xorbApiKey}`,
            'Content-Type': 'application/json'
          }
        })
      );

      // Process and validate AI response
      return this.processAttackChainResponse(data);
    } catch (error) {
      this.handleError(error, 'Attack chain detection');
      return [];
    }
  }

  /**
   * Detect potential false positives
   * @param vulnerabilityIds Array of vulnerability IDs
   * @param tenantId ID of the tenant for context
   * @returns False positive detection results
   */
  async detectFalsePositives(vulnerabilityIds: string[], tenantId: string): Promise<FalsePositiveDetection[]> {
    const vulnerabilities = await this.vulnerabilityRepository.findByIds(vulnerabilityIds);
    
    const payload = {
      model: this.xorbAiModel,
      input: {
        vulnerabilities: vulnerabilities.map(vuln => ({
          id: vuln.id,
          title: vuln.title,
          description: vuln.description,
          technicalDetails: vuln.technicalDetails,
          cvssScore: vuln.cvssScore,
          assetContext: vuln.assetContext
        })),
        tenantContext: await this.getTenantContext(tenantId)
      }
    };

    try {
      const { data } = await firstValueFrom(
        this.httpService.post(`${this.xorbApiUrl}/false-positive-detection`, payload, {
          headers: { 
            'Authorization': `Bearer ${this.xorbApiKey}`,
            'Content-Type': 'application/json'
          }
        })
      );

      // Process and validate AI response
      return this.processFalsePositiveDetectionResponse(data);
    } catch (error) {
      this.handleError(error, 'False positive detection');
      return [];
    }
  }

  /**
   * Process analyst feedback on false positives to improve detection
   * @param vulnerabilityId ID of the vulnerability
   * @param isFalsePositive Whether the vulnerability is a false positive
   * @param analystNotes Analyst's explanation
   * @returns Success status
   */
  async processFalsePositiveFeedback(vulnerabilityId: string, isFalsePositive: boolean, analystNotes: string): Promise<boolean> {
    const vulnerability = await this.vulnerabilityRepository.findOneBy({ id: vulnerabilityId });
    
    const payload = {
      model: this.xorbAiModel,
      input: {
        vulnerability: {
          id: vulnerability.id,
          title: vulnerability.title,
          description: vulnerability.description,
          technicalDetails: vulnerability.technicalDetails
        },
        feedback: {
          isFalsePositive,
          analystNotes,
          timestamp: new Date().toISOString()
        }
      }
    };

    try {
      const { data } = await firstValueFrom(
        this.httpService.post(`${this.xorbApiUrl}/false-positive-feedback`, payload, {
          headers: { 
            'Authorization': `Bearer ${this.xorbApiKey}`,
            'Content-Type': 'application/json'
          }
        })
      );

      return data.success;
    } catch (error) {
      this.handleError(error, 'False positive feedback processing');
      return false;
    }
  }

  /**
   * Get tenant context for AI analysis
   * @param tenantId ID of the tenant
   * @returns Tenant context
   */
  private async getTenantContext(tenantId: string): Promise<any> {
    // In a real implementation, this would fetch tenant-specific context
    // from the database or configuration service
    return {
      id: tenantId,
      industry: 'technology',
      complianceRequirements: ['GDPR', 'ISO27001'],
      assetCriticalityWeights: {
        publicFacing: 0.8,
        internal: 0.5,
        legacy: 0.6
      }
    };
  }

  /**
   * Process and validate risk priority scores response from AI
   * @param data Raw AI response
   * @returns Processed risk priority scores
   */
  private processRiskPriorityScoresResponse(data: any): RiskPriorityScore[] {
    // In a real implementation, this would validate and process the AI response
    // against a schema and enrich with additional data
    return data.scores.map(score => ({
      vulnerabilityId: score.vulnerabilityId,
      score: score.calculatedScore,
      breakdown: score.breakdown,
      explanation: score.explanation,
      confidence: score.confidenceScore,
      timestamp: new Date()
    }));
  }

  /**
   * Process and validate remediation guidance response from AI
   * @param data Raw AI response
   * @returns Processed remediation suggestions
   */
  private processRemediationGuidanceResponse(data: any): RemediationSuggestion[] {
    // In a real implementation, this would validate and process the AI response
    // against a schema and enrich with additional data
    return data.remediations.map(remediation => ({
      vulnerabilityId: remediation.vulnerabilityId,
      technicalSteps: remediation.technicalSteps,
      executiveSummary: remediation.executiveSummary,
      implementationTime: remediation.estimatedTimeMinutes,
      complexity: remediation.complexityLevel,
      confidence: remediation.confidenceScore,
      timestamp: new Date()
    }));
  }

  /**
   * Process and validate attack chain response from AI
   * @param data Raw AI response
   * @returns Processed attack chains
   */
  private processAttackChainResponse(data: any): AttackChain[] {
    // In a real implementation, this would validate and process the AI response
    // against a schema and enrich with additional data
    return data.attackChains.map(chain => ({
      id: chain.id,
      name: chain.name,
      description: chain.description,
      steps: chain.steps.map(step => ({
        vulnerabilityId: step.vulnerabilityId,
        order: step.order,
        description: step.description
      })),
      impact: chain.impactLevel,
      confidence: chain.confidenceScore,
      timestamp: new Date()
    }));
  }

  /**
   * Process and validate false positive detection response from AI
   * @param data Raw AI response
   * @returns Processed false positive detection results
   */
  private processFalsePositiveDetectionResponse(data: any): FalsePositiveDetection[] {
    // In a real implementation, this would validate and process the AI response
    // against a schema and enrich with additional data
    return data.detections.map(detection => ({
      vulnerabilityId: detection.vulnerabilityId,
      isLikelyFalsePositive: detection.isLikelyFalsePositive,
      confidence: detection.confidenceScore,
      explanation: detection.explanation,
      timestamp: new Date()
    }));
  }

  /**
   * Handle errors from AI API calls
   * @param error Error object
   * @param operation Operation name for logging
   */
  private handleError(error: any, operation: string): void {
    // In a real implementation, this would include more sophisticated error handling,
    // logging, and possibly fallback mechanisms
    console.error(`Error during ${operation}:`, {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message
    });
    
    // You might want to emit an event or trigger a monitoring alert here
  }
}