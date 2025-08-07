import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import { WebhookTriggerDto } from '../../dto/webhook-trigger.dto';
import { IntegrationConfig } from '../../interfaces/integration-config.interface';

@Injectable()
export class TeamsBotTriggerService {
  constructor(private readonly httpService: HttpService) {}

  async triggerScan(config: IntegrationConfig, triggerData: WebhookTriggerDto): Promise<void> {
    try {
      const response = await firstValueFrom(
        this.httpService.post(config.webhookUrl, {
          '@type': 'MessageCard',
          '@context': 'https://schema.org/extensions',
          'themeColor': '0078D7',
          'summary': 'PTaaS Scan Triggered',
          'sections': [{
            'activityTitle': 'New scan triggered via Microsoft Teams',
            'activitySubtitle': `Asset: ${triggerData.assetName} | Type: ${triggerData.scanType}`,
            'facts': [
              { 'name': 'Scan ID', 'value': triggerData.scanId },
              { 'name': 'Triggered by', 'value': triggerData.triggeredBy },
              { 'name': 'Timestamp', 'value': new Date().toISOString() }
            ],
            'markdown': true
          }],
          'potentialAction': [{
            'name': 'View Scan Details',
            '@type': 'OpenUri',
            'targets': [{ 'os': 'default', 'uri': `${process.env.FRONTEND_URL}/scans/${triggerData.scanId}` }]
          }]
        }]
      })
      );

      if (response.status !== 200) {
        throw new Error(`Failed to trigger Teams notification: ${response.statusText}`);
      }
    } catch (error) {
      throw new Error(`Teams bot trigger error: ${error.message}`);
    }
  }

  async handleScanResults(config: IntegrationConfig, resultsData: any): Promise<void> {
    try {
      const response = await firstValueFrom(
        this.httpService.post(config.webhookUrl, {
          '@type': 'MessageCard',
          '@context': 'https://schema.org/extensions',
          'themeColor': 'FF0000',
          'summary': 'PTaaS Scan Results Available',
          'sections': [{
            'activityTitle': 'Scan results received via Microsoft Teams',
            'activitySubtitle': `Asset: ${resultsData.assetName} | Scan ID: ${resultsData.scanId}`,
            'facts': [
              { 'name': 'Severity Count', 'value': `Critical: ${resultsData.criticalCount}, High: ${resultsData.highCount}` },
              { 'name': 'Compliance Issues', 'value': resultsData.complianceIssues.join(', ') },
              { 'name': 'Timestamp', 'value': new Date().toISOString() }
            ],
            'markdown': true
          }],
          'potentialAction': [{
            'name': 'View Results',
            '@type': 'OpenUri',
            'targets': [{ 'os': 'default', 'uri': `${process.env.FRONTEND_URL}/reports/${resultsData.reportId}` }]
          }]
        }]
      })
      );

      if (response.status !== 200) {
        throw new Error(`Failed to send results to Teams: ${response.statusText}`);
      }
    } catch (error) {
      throw new Error(`Teams bot results error: ${error.message}`);
    }
  }
}