import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { IntegrationBaseService } from './integration.base.service';
import { ScanResult } from '../../scans/scan-result.entity';

@Injectable()
export class ServiceNowIntegrationService extends IntegrationBaseService {
  constructor(
    protected readonly httpService: HttpService,
    protected readonly configService: ConfigService
  ) {
    super(httpService, configService, 'servicenow');
  }

  async createIncident(scanResult: ScanResult): Promise<any> {
    const incidentData = {
      short_description: `Security Vulnerability: ${scanResult.vulnerability.name}`,
      description: this.formatVulnerabilityDescription(scanResult),
      urgency: this.mapSeverityToUrgency(scanResult.vulnerability.severity),
      impact: this.mapSeverityToImpact(scanResult.vulnerability.severity),
      category: 'security',
      subcategory: scanResult.scanType,
    };

    return this.makeApiRequest('/api/now/table/incident', 'post', incidentData);
  }

  async updateIncident(incidentId: string, updateData: any): Promise<any> {
    return this.makeApiRequest(`/api/now/table/incident/${incidentId}`, 'put', updateData);
  }

  async getIncident(incidentId: string): Promise<any> {
    return this.makeApiRequest(`/api/now/table/incident/${incidentId}`);
  }

  private formatVulnerabilityDescription(scanResult: ScanResult): string {
    return `
Vulnerability: ${scanResult.vulnerability.name}
Severity: ${scanResult.vulnerability.severity}
Description: ${scanResult.vulnerability.description}

Impact: ${scanResult.vulnerability.impact}

Remediation: ${scanResult.vulnerability.remediation}

Technical Details:
${JSON.stringify(scanResult.details, null, 2)}
    `;
  }

  private mapSeverityToUrgency(severity: string): number {
    switch (severity.toLowerCase()) {
      case 'critical': return 1;
      case 'high': return 2;
      case 'medium': return 3;
      case 'low': return 4;
      default: return 3;
    }
  }

  private mapSeverityToImpact(severity: string): number {
    switch (severity.toLowerCase()) {
      case 'critical': return 1;
      case 'high': return 2;
      case 'medium': return 3;
      case 'low': return 4;
      default: return 3;
    }
  }

  protected getAuthConfig() {
    return {
      username: this.configService.get('SERVICENOW_USERNAME'),
      password: this.configService.get('SERVICENOW_PASSWORD'),
    };
  }

  protected getApiBaseUrl(): string {
    return this.configService.get('SERVICENOW_INSTANCE_URL');
  }
}
