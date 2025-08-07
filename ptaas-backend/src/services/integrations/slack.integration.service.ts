import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';
import { ScanRequest } from '../../scanning/dto/scan-request.dto';
import { ScanResult } from '../../scanning/dto/scan-result.dto';

@Injectable()
export class SlackIntegrationService {
  private slackWebhookUrl: string;

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService
  ) {
    this.slackWebhookUrl = this.configService.get<string>('SLACK_WEBHOOK_URL');
  }

  async sendScanRequestNotification(scanRequest: ScanRequest, userId: string): Promise<void> {
    if (!this.slackWebhookUrl) {
      return; // Skip if not configured
    }

    const message = {
      text: `New security scan requested by user ${userId}`,
      blocks: [
        {
          type: "header",
          text: {
            type: "plain_text",
            text: "🔐 Security Scan Requested"
          }
        },
        {
          type: "section",
          text: {
            type: "mrkdwn",
            text: `*Target:* ${scanRequest.target}\n*Scan Type:* ${scanRequest.scanType}\n*User:* ${userId}`
          }
        }
      ]
    };

    try {
      await firstValueFrom(this.httpService.post(this.slackWebhookUrl, message));
    } catch (error) {
      console.error('Failed to send Slack notification:', error.message);
    }
  }

  async sendScanCompletionNotification(scanResult: ScanResult): Promise<void> {
    if (!this.slackWebhookUrl) {
      return; // Skip if not configured
    }

    const severityColor = this.getSeverityColor(scanResult.overallSeverity);
    const message = {
      text: `Security scan completed for ${scanResult.target} with ${scanResult.vulnerabilities.length} vulnerabilities`,
      blocks: [
        {
          type: "header",
          text: {
            type: "plain_text",
            text: "📊 Scan Results Available"
          }
        },
        {
          type: "section",
          text: {
            type: "mrkdwn",
            text: `*Target:* ${scanResult.target}\n*Scan Type:* ${scanResult.scanType}\n*Vulnerabilities Found:* ${scanResult.vulnerabilities.length}\n*Overall Severity:* ${scanResult.overallSeverity}`
          }
        },
        {
          type: "divider"
        },
        {
          type: "section",
          text: {
            type: "mrkdwn",
            text: "View full report in the PTaaS platform"
          },
          accessory: {
            type: "button",
            text: {
              type: "plain_text",
              text: "View Report"
            },
            url: `${this.configService.get<string>('APP_URL')}/reports/${scanResult.id}`
          }
        }
      ],
      color: severityColor
    };

    try {
      await firstValueFrom(this.httpService.post(this.slackWebhookUrl, message));
    } catch (error) {
      console.error('Failed to send Slack notification:', error.message);
    }
  }

  private getSeverityColor(severity: string): string {
    switch (severity.toLowerCase()) {
      case 'critical':
        return '#ff403d'; // Red
      case 'high':
        return '#ff6f61'; // Orange-Red
      case 'medium':
        return '#ffcc00'; // Yellow
      case 'low':
        return '#66cc66'; // Green
      default:
        return '#999999'; // Gray
    }
  }
}