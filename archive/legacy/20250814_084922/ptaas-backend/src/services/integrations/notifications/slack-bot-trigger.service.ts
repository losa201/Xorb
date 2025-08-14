import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { lastValueFrom } from 'rxjs';
import { SlackMessage } from './dto/slack-message.dto';
import { ScanTriggerEvent } from '../scanning/dto/scan-trigger-event.dto';

@Injectable()
export class SlackBotTriggerService {
  private readonly slackWebhookUrl: string;

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService
  ) {
    this.slackWebhookUrl = this.configService.get<string>('SLACK_WEBHOOK_URL');
  }

  async triggerScan(message: SlackMessage): Promise<ScanTriggerEvent> {
    // Verify the message came from Slack using the signing secret
    const isValidRequest = this.verifySlackRequest(message);
    if (!isValidRequest) {
      throw new Error('Invalid Slack request signature');
    }

    // Parse the slash command
    if (message.text.startsWith('scan ')) {
      const asset = message.text.substring(5).trim();

      // Create scan trigger event
      const scanEvent: ScanTriggerEvent = {
        asset,
        initiatedBy: `slack:${message.user_id}`,
        timestamp: new Date(),
        scanType: this.determineScanType(asset),
        options: this.parseScanOptions(message)
      };

      // Trigger the scan via the scan orchestration service
      // This would typically publish an event to a message queue
      // For simplicity, we'll return the scan event to be handled by the orchestrator
      return scanEvent;
    } else if (message.text.startsWith('help')) {
      await this.sendHelpMessage(message.response_url);
    } else {
      await this.sendUnknownCommandMessage(message.response_url);
    }
  }

  async sendScanResults(scanId: string, results: any, webhookUrl: string): Promise<void> {
    const summary = this.generateScanSummary(results);

    const message = {
      text: `Security Scan Results for ${scanId}`,
      blocks: [
        {
          type: 'header',
          text: {
            type: 'plain_text',
            text: `🔍 Security Scan Results for ${scanId}`,
            emoji: true
          }
        },
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: summary
          }
        },
        {
          type: 'actions',
          elements: [
            {
              type: 'button',
              text: {
                type: 'plain_text',
                text: 'View Full Report',
                emoji: true
              },
              value: scanId,
              url: `${this.configService.get<string>('APP_URL')}/reports/${scanId}`
            }
          ]
        }
      ]
    };

    await lastValueFrom(this.httpService.post(webhookUrl, message));
  }

  private verifySlackRequest(message: any): boolean {
    // In a real implementation, we would verify the request signature using the signing secret
    // This is a simplified version for demonstration purposes
    return !!message.user_id && !!message.text;
  }

  private determineScanType(asset: string): string {
    // In a real implementation, we would use more sophisticated logic
    // to determine the scan type based on the asset
    if (asset.startsWith('http://') || asset.startsWith('https://')) {
      return 'web';
    } else if (asset.includes('api.') || asset.includes('/api/')) {
      return 'api';
    } else if (asset.includes('.cloud') || asset.includes('.aws') || asset.includes('.azure') || asset.includes('.gcp')) {
      return 'cloud';
    } else {
      return 'web'; // Default to web scan
    }
  }

  private parseScanOptions(message: any): any {
    // Parse any additional scan options from the message
    // This is a simplified version
    return {
      priority: 'normal',
      scanType: this.determineScanType(message.text)
    };
  }

  private generateScanSummary(results: any): string {
    // Generate a summary of the scan results
    const vulnerabilities = results.vulnerabilities || [];
    const highSeverity = vulnerabilities.filter(v => v.severity === 'high').length;
    const mediumSeverity = vulnerabilities.filter(v => v.severity === 'medium').length;

    return `*Scan completed with ${vulnerabilities.length} vulnerabilities found:*
• ${highSeverity} High severity
• ${mediumSeverity} Medium severity
• ${vulnerabilities.length - highSeverity - mediumSeverity} Low severity

Overall risk score: ${results.riskScore || 'N/A'}`;
  }

  private async sendHelpMessage(webhookUrl: string): Promise<void> {
    const message = {
      text: 'Available commands',
      blocks: [
        {
          type: 'header',
          text: {
            type: 'plain_text',
            text: '🤖 PTaaS Slack Bot Help',
            emoji: true
          }
        },
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: 'Available commands:\n`/ptaaS scan [asset]` - Start a security scan for an asset\n`/ptaaS help` - Show this help message'
          }
        }
      ]
    };

    await lastValueFrom(this.httpService.post(webhookUrl, message));
  }

  private async sendUnknownCommandMessage(webhookUrl: string): Promise<void> {
    const message = {
      text: 'Unknown command',
      blocks: [
        {
          type: 'header',
          text: {
            type: 'plain_text',
            text: '🤖 PTaaS Slack Bot',
            emoji: true
          }
        },
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: 'Unknown command. Type `/ptaaS help` for available commands.'
          }
        }
      ]
    };

    await lastValueFrom(this.httpService.post(webhookUrl, message));
  }
}
