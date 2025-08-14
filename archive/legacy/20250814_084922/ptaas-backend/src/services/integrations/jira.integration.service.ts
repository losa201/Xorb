import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';
import { IntegrationConfig, JiraConfig, JiraIssue, JiraProject } from '../../interfaces/integration.interface';

@Injectable()
export class JiraIntegrationService {
  private readonly jiraApiUrl: string;

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService
  ) {
    const jiraConfig = this.configService.get<JiraConfig>('jira');
    this.jiraApiUrl = `${jiraConfig.host}/rest/api/3`;
  }

  async getProjects(): Promise<JiraProject[]> {
    const jiraAuth = this.getJiraAuth();
    const { data } = await firstValueFrom(
      this.httpService.get(`${this.jiraApiUrl}/project/search`, {
        headers: {
          'Authorization': `Basic ${jiraAuth}`,
          'Accept': 'application/json',
        },
      })
    );
    return data.values;
  }

  async createIssue(issue: JiraIssue): Promise<any> {
    const jiraAuth = this.getJiraAuth();
    const { data } = await firstValueFrom(
      this.httpService.post(`${this.jiraApiUrl}/issue`, issue, {
        headers: {
          'Authorization': `Basic ${jiraAuth}`,
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
      })
    );
    return data;
  }

  async getIssue(issueIdOrKey: string): Promise<any> {
    const jiraAuth = this.getJiraAuth();
    const { data } = await firstValueFrom(
      this.httpService.get(`${this.jiraApiUrl}/issue/${issueIdOrKey}`, {
        headers: {
          'Authorization': `Basic ${jiraAuth}`,
          'Accept': 'application/json',
        },
      })
    );
    return data;
  }

  async updateIssue(issueIdOrKey: string, updateData: any): Promise<any> {
    const jiraAuth = this.getJiraAuth();
    const { data } = await firstValueFrom(
      this.httpService.put(`${this.jiraApiUrl}/issue/${issueIdOrKey}`, updateData, {
        headers: {
          'Authorization': `Basic ${jiraAuth}`,
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
      })
    );
    return data;
  }

  async searchIssues(jql: string): Promise<any> {
    const jiraAuth = this.getJiraAuth();
    const { data } = await firstValueFrom(
      this.httpService.post(`${this.jiraApiUrl}/search`, {
        jql,
        maxResults: 100
      }, {
        headers: {
          'Authorization': `Basic ${jiraAuth}`,
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
      })
    );
    return data;
  }

  private getJiraAuth(): string {
    const jiraConfig = this.configService.get<JiraConfig>('jira');
    return Buffer.from(`${jiraConfig.user}:${jiraConfig.apiToken}`).toString('base64');
  }

  async testConnection(): Promise<boolean> {
    try {
      const jiraAuth = this.getJiraAuth();
      const { status } = await firstValueFrom(
        this.httpService.get(`${this.jiraApiUrl}/serverInfo`, {
          headers: {
            'Authorization': `Basic ${jiraAuth}`,
            'Accept': 'application/json',
          },
        })
      );
      return status === 200;
    } catch (error) {
      return false;
    }
  }

  async getIssueTypes(projectId: string): Promise<any> {
    const jiraAuth = this.getJiraAuth();
    const { data } = await firstValueFrom(
      this.httpService.get(`${this.jiraApiUrl}/project/${projectId}/issuetypes`, {
        headers: {
          'Authorization': `Basic ${jiraAuth}`,
          'Accept': 'application/json',
        },
      })
    );
    return data;
  }

  async getPriorities(): Promise<any> {
    const jiraAuth = this.getJiraAuth();
    const { data } = await firstValueFrom(
      this.httpService.get(`${this.jiraApiUrl}/priority`, {
        headers: {
          'Authorization': `Basic ${jiraAuth}`,
          'Accept': 'application/json',
        },
      })
    );
    return data;
  }

  async handleScanResult(scanResult: any): Promise<void> {
    // Implementation for handling scan results and creating Jira issues
    // This would be called from the scan orchestration service
    const projectKey = this.configService.get('jira.defaultProjectKey');
    const issueType = this.configService.get('jira.defaultIssueType');

    // Create Jira issue for each critical vulnerability
    for (const vulnerability of scanResult.criticalVulnerabilities) {
      const jiraIssue = this.mapVulnerabilityToJiraIssue(vulnerability, projectKey, issueType);
      await this.createIssue(jiraIssue);
    }
  }

  private mapVulnerabilityToJiraIssue(vulnerability: any, projectKey: string, issueType: string): JiraIssue {
    return {
      fields: {
        project: {
          key: projectKey
        },
        summary: `${vulnerability.type}: ${vulnerability.title}`,
        description: {
          version: 1,
          type: 'doc',
          content: [
            {
              type: 'paragraph',
              content: [
                { type: 'text', text: 'Vulnerability Details:' }
              ]
            },
            { type: 'hardBreak' },
            {
              type: 'paragraph',
              content: [
                { type: 'text', text: `Severity: ${vulnerability.severity}` }
              ]
            },
            { type: 'hardBreak' },
            {
              type: 'paragraph',
              content: [
                { type: 'text', text: `Description: ${vulnerability.description}` }
              ]
            },
            { type: 'hardBreak' },
            {
              type: 'paragraph',
              content: [
                { type: 'text', text: `Recommendation: ${vulnerability.recommendation}` }
              ]
            }
          ]
        },
        issuetype: {
          name: issueType
        },
        priority: {
          name: this.mapSeverityToJiraPriority(vulnerability.severity)
        }
      }
    };
  }

  private mapSeverityToJiraPriority(severity: string): string {
    switch (severity.toLowerCase()) {
      case 'critical':
        return 'Highest';
      case 'high':
        return 'High';
      case 'medium':
        return 'Medium';
      case 'low':
        return 'Low';
      default:
        return 'Medium';
    }
  }
}
