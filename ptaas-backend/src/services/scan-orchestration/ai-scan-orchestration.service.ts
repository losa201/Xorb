import { XORB_API } from '../../config/xorb-api';
import { ScanRequest, ScanType, ScanSequence, AssetProfile } from '../../models/scan.model';
import { ThreatIntelService } from '../threat-intel/threat-intel.service';
import { LocalizationService } from '../localization/localization.service';
import { Vulnerability } from '../../models/vulnerability.model';

interface AIOrchestrationResponse {
  scanTypes: ScanType[];
  sequence: ScanSequence;
  priority: number;
  reasoning: string;
}

interface RemediationGuidance {
  summary: string;
  steps: string[];
  resources: string[];
  localizedContent: {
    [key: string]: {
      summary: string;
      steps: string[];
    };
  };
}

export class AIScanOrchestrationService {
  private xorBClient: XORB_API;
  private threatIntel: ThreatIntelService;
  private localization: LocalizationService;

  constructor() {
    this.xorBClient = new XORB_API(process.env.XORB_API_KEY);
    this.threatIntel = new ThreatIntelService();
    this.localization = new LocalizationService();
  }

  async determineScanStrategy(asset: AssetProfile): Promise<ScanSequence> {
    try {
      // Get live threat intelligence
      const threatData = await this.threatIntel.getThreatIntelForAsset(asset);
      
      // Prepare AI request
      const aiRequest = {
        assetProfile: this.prepareAssetProfileForAI(asset),
        threatIntel: threatData,
        currentDateTime: new Date().toISOString()
      };

      // Get AI recommendations from XORB API
      const aiResponse = await this.xorBClient.post<AIOrchestrationResponse>(
        '/ai/orchestration/scan-strategy',
        aiRequest
      );

      // Process and validate AI response
      const validatedSequence = this.validateAndAdjustSequence(
        aiResponse.data,
        asset.supportedScanTypes
      );

      // Store AI reasoning for audit and reporting
      this.storeAIDecisionMetadata(asset.id, aiResponse.data.reasoning);

      return validatedSequence;
    } catch (error) {
      // Fallback to default scan strategy on AI failure
      console.error(`AI orchestration failed: ${error.message}`);
      return this.getDefaultScanSequence(asset);
    }
  }

  async generateRemediationGuidance(vulnerability: Vulnerability): Promise<RemediationGuidance> {
    try {
      // Get base guidance from XORB AI
      const aiGuidance = await this.xorBClient.post<RemediationGuidance>(
        '/ai/remediation/generate',
        {
          vulnerabilityId: vulnerability.id,
          severity: vulnerability.severity,
          assetType: vulnerability.assetType,
          environment: vulnerability.environment
        }
      );

      // Localize content for EN/DE
      const localizedGuidance = await this.localization.translateRemediation(
        aiGuidance.data,
        ['en', 'de']
      );

      return {
        ...aiGuidance.data,
        localizedContent: localizedGuidance
      };
    } catch (error) {
      console.error(`AI remediation generation failed: ${error.message}`);
      return this.getDefaultRemediation(vulnerability);
    }
  }

  private prepareAssetProfileForAI(asset: AssetProfile): any {
    // Prepare asset profile in format required by AI model
    return {
      id: asset.id,
      type: asset.type,
      technologies: asset.technologies,
      criticality: asset.criticality,
      complianceRequirements: asset.complianceRequirements,
      previousFindings: asset.previousFindings,
      networkExposure: asset.networkExposure
    };
  }

  private validateAndAdjustSequence(
    aiResponse: AIOrchestrationResponse,
    supportedTypes: ScanType[]
  ): ScanSequence {
    // Filter out unsupported scan types
    const validScanTypes = aiResponse.scanTypes.filter(
      type => supportedTypes.includes(type)
    );

    // Ensure at least one scan type is selected
    if (validScanTypes.length === 0) {
      validScanTypes.push(supportedTypes[0]);
    }

    // Create sequence with timing and dependencies
    return {
      scanTypes: validScanTypes,
      order: this.calculateOptimalOrder(validScanTypes, aiResponse.sequence),
      timingConstraints: this.calculateTimingConstraints(validScanTypes),
      dependencies: this.calculateDependencies(validScanTypes),
      priority: aiResponse.priority
    };
  }

  private calculateOptimalOrder(
    scanTypes: ScanType[],
    aiSequence: ScanSequence
  ): string[] {
    // Implement logic to determine optimal scan order based on:
    // - Scan type dependencies
    // - Asset profile
    // - Threat intelligence
    // - AI recommendations
    
    // This is a simplified example - actual implementation would be more complex
    const order = [...scanTypes];
    
    // Prioritize reconnaissance scans first
    if (order.includes('recon')) {
      const reconIndex = order.indexOf('recon');
      if (reconIndex > 0) {
        order.splice(reconIndex, 1);
        order.unshift('recon');
      }
    }
    
    // Put high-risk scans later
    if (order.includes('exploitation')) {
      const exploitIndex = order.indexOf('exploitation');
      if (exploitIndex < order.length - 1) {
        order.splice(exploitIndex, 1);
        order.push('exploitation');
      }
    }
    
    return order.map(scan => scan.toString());
  }

  private calculateTimingConstraints(scanTypes: ScanType[]): any {
    // Calculate optimal timing based on scan types and asset characteristics
    return {
      maxDuration: this.calculateMaxDuration(scanTypes),
      optimalWindow: this.calculateOptimalWindow(scanTypes),
      concurrency: this.calculateConcurrency(scanTypes)
    };
  }

  private calculateMaxDuration(scanTypes: ScanType[]): number {
    // Calculate estimated maximum duration based on scan types
    const durationMap: Record<ScanType, number> = {
      recon: 30,
      vulnerability: 60,
      exploitation: 90,
      post_exploitation: 45,
      api: 75,
      cloud: 120,
      container: 90
    };

    return scanTypes.reduce((total, type) => total + (durationMap[type] || 60), 0);
  }

  private calculateOptimalWindow(scanTypes: ScanType[]): { start: string; end: string } {
    // Determine optimal scan window based on scan types
    // For example, infrastructure scans during off-peak hours
    return {
      start: '22:00',
      end: '06:00'
    };
  }

  private calculateConcurrency(scanTypes: ScanType[]): number {
    // Determine optimal concurrency level based on scan types
    // More resource-intensive scans get lower concurrency
    return scanTypes.some(type => type === 'exploitation' || type === 'container') ? 2 : 5;
  }

  private calculateDependencies(scanTypes: ScanType[]): Record<string, string[]> {
    // Define dependencies between scan types
    const dependencies: Record<string, string[]> = {};
    
    if (scanTypes.includes('exploitation') && scanTypes.includes('recon')) {
      dependencies['exploitation'] = ['recon'];
    }
    
    if (scanTypes.includes('post_exploitation') && scanTypes.includes('exploitation')) {
      dependencies['post_exploitation'] = ['exploitation'];
    }
    
    return dependencies;
  }

  private storeAIDecisionMetadata(assetId: string, reasoning: string): void {
    // Store AI decision metadata for audit and future reference
    // This would typically be stored in a database
    console.log(`AI decision for asset ${assetId}: ${reasoning}`);
  }

  private getDefaultScanSequence(asset: AssetProfile): ScanSequence {
    // Fallback scan sequence when AI is unavailable
    return {
      scanTypes: [asset.defaultScanType || 'vulnerability'],
      order: [asset.defaultScanType || 'vulnerability'],
      timingConstraints: {
        maxDuration: 60,
        optimalWindow: { start: '22:00', end: '06:00' },
        concurrency: 3
      },
      dependencies: {}
    };
  }

  private getDefaultRemediation(vulnerability: Vulnerability): RemediationGuidance {
    // Fallback remediation guidance when AI is unavailable
    return {
      summary: `Standard remediation for ${vulnerability.type} vulnerability`,
      steps: [
        'Apply vendor patches',
        'Update configurations',
        'Verify fix with follow-up scan'
      ],
      resources: [
        'https://cve.mitre.org/',
        'https://nvd.nist.gov/'
      ],
      localizedContent: {
        en: {
          summary: `Standard remediation for ${vulnerability.type} vulnerability`,
          steps: [
            'Apply vendor patches',
            'Update configurations',
            'Verify fix with follow-up scan'
          ]
        },
        de: {
          summary: `Standardmaßnahmen zur Behebung von ${vulnerability.type}-Schwachstellen`,
          steps: [
            'Anwenden von Herstellerpatches',
            'Aktualisieren von Konfigurationen',
            'Überprüfung der Behebung mit Folgescans'
          ]
        }
      }
    };
  }
}