import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';
import { IntegrationService } from '../integration.service';
import { GithubActionsWebhookDto } from './dto/github-actions-webhook.dto';

@Injectable()
export class GithubActionsIntegrationService {
  constructor(
    private readonly configService: ConfigService,
    private readonly httpService: HttpService,
    private readonly integrationService: IntegrationService,
  ) {}

  async triggerScanWorkflow(repository: string, branch: string, scanType: string): Promise<void> {
    const githubToken = this.configService.get<string>('GITHUB_TOKEN');
    const workflowId = this.configService.get<string>('GITHUB_SCAN_WORKFLOW_ID');

    try {
      const response = await firstValueFrom(
        this.httpService.post(
          `https://api.github.com/repos/${repository}/actions/workflows/${workflowId}/dispatches`,
          {
            ref: branch,
            inputs: {
              scanType: scanType,
              ptaasApiKey: this.configService.get<string>('PTAAS_API_KEY'),
              ptaasApiUrl: this.configService.get<string>('PTAAS_API_URL')
            }
          },
          {
            headers: {
              Authorization: `Bearer ${githubToken}`,
              Accept: 'application/vnd.github+json',
              'X-GitHub-Api-Version': '2022-11-28'
            }
          }
        )
      );

      if (response.status !== 204) {
        throw new Error(`Failed to trigger GitHub workflow: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Error triggering GitHub workflow:', error);
      throw error;
    }
  }

  async handleWebhook(dto: GithubActionsWebhookDto): Promise<void> {
    // Handle GitHub webhook events
    // This would be used to receive scan results or status updates from GitHub Actions
  }
}
