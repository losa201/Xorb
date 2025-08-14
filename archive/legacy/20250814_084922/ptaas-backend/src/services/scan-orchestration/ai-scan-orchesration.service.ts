import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { lastValueFrom } from 'rxjs';
import { ScanRequestDto } from '../dto/scan-request.dto';
import { ScanPlan } from '../interfaces/scan-plan.interface';
import { Vulnerability } from '../interfaces/vulnerability.interface';

@Injectable()
export class AiScanOrchestrationService {
  private readonly xorbaseUrl: string;
  private readonly apiKey: string;

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService
  ) {
    this.xorbaseUrl = this.configService.get<string>('XORB_API_URL');
    this.apiKey = this.configService.get<string>('XORB_API_KEY');
  }

  async generateScanPlan(scanRequest: ScanRequestDto): Promise<ScanPlan> {
    const { data } = await lastValueFrom(
      this.httpService.post(`${this.xorbaseUrl}/v1/ai/scan-planning`, {
        asset_profile: scanRequest.assetProfile,
        threat_intel: scanRequest.threatIntel,
        compliance_requirements: scanRequest.complianceRequirements
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      })
    );

    return {
      scan_sequence: data.scan_sequence,
      priority_scores: data.priority_scores,
      justification: data.justification,
      threat_intel_sources: data.threat_intel_sources
    };
  }

  async generateRemediationGuidance(vulnerabilities: Vulnerability[], language: string = 'en') {
    const { data } = await lastValueFrom(
      this.httpService.post(`${this.xorbaseUrl}/v1/ai/remediation`, {
        vulnerabilities,
        language
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      })
    );

    return data.remediation_guidance;
  }

  async generateExplanationOverlay(vulnerabilityId: string, language: string = 'en') {
    const { data } = await lastValueFrom(
      this.httpService.get(`${this.xorbaseUrl}/v1/ai/explanation/${vulnerabilityId}`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
        params: {
          language
        }
      })
    );

    return data.explanation;
  }
}
