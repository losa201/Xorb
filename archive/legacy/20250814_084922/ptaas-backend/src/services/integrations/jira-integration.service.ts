import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import { IntegrationConfig } from '../../interfaces/integration-config.interface';
import { Vulnerability } from '../../interfaces/vulnerability.interface';

@Injectable()
export class JiraIntegrationService {
  private jiraApiUrl: string;
  private jiraUser: string;
  private jiraApiToken: string;

  constructor(
    private configService: ConfigService,
    private httpService: HttpService
  ) {
    const config = this.configService.get<IntegrationConfig>('jira');
    this.jiraApiUrl = config.apiUrl;
    this.jiraUser = config.user;
    this.jiraApiToken = config.apiToken;
  }

  async createVulnerabilityIssue(vulnerability: Vulnerability, projectId: string, issueType: string): Promise<any> {
    const auth = Buffer.from(`${this.jiraUser}:${this.jiraApiToken}`).toString('base64');

    const issueData = {
      fields: {
        project: {
          id: projectId
        },
        summary: `[AUTO] ${vulnerability.title} - ${vulnerability.severity.toUpperCase()}`,
        description: {
          version: 1,
          type: 'doc',
          content: [
            {
              type: 'paragraph',
              content: [
                { type: 'text', text: 'Automatically created vulnerability issue from PTaaS platform' }
              ]
            },
            {
              type: 'heading',
              attrs: { level: 2 },
              content: [{ type: 'text', text: 'Vulnerability Details' }]
            },
            {
              type: 'paragraph',
              content: [
                { type: 'text', text: `Severity: ${vulnerability.severity.toUpperCase()}` }
              ]
            },
            {
              type: 'paragraph',
              content: [
                { type: 'text', text: `CVSS Score: ${vulnerability.cvssScore || 'N/A'}` }
              ]
            },
            {
              type: 'paragraph',
              content: [
                { type: 'text', text: `Asset: ${vulnerability.asset}` }
              ]
            },
            {
              type: 'paragraph',
              content: [
                { type: 'text', text: `Description: ${vulnerability.description}` }
              ]
            },
            {
              type: 'paragraph',
              content: [
                { type: 'text', text: `Remediation: ${vulnerability.remediation}` }
              ]
            }
          ]
        },
        issuetype: {
          name: issueType
        }
      }
    };

    try {
      const response = await firstValueFrom(
        this.httpService.post(`${this.jiraApiUrl}/rest/api/3/issue`, issueData, {
          headers: {
            'Authorization': `Basic ${auth}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          }
        })
      );
      return response.data;
    } catch (error) {
      console.error('Error creating Jira issue:', error.response?.data || error.message);
      throw error;
    }
  }

  async getProjects(): Promise<any> {
    const auth = Buffer.from(`${this.jiraUser}:${this.jiraApiToken}`).toString('base64');

    try {
      const response = await firstValueFrom(
        this.httpService.get(`${this.jiraApiUrl}/rest/api/3/project`, {
          headers: {
            'Authorization': `Basic ${auth}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          }
        })
      );
      return response.data;
    } catch (error) {
      console.error('Error fetching Jira projects:', error.response?.data || error.message);
      throw error;
    }
  }

  async getIssueTypes(): Promise<any> {
    try {
      const auth = Buffer.from(`${this.jiraUser}:${this.jiraApiToken}`).toString('base64');

      const response = await firstValueFrom(
        this.httpService.get(`${this.jiraApiUrl}/rest/api/3/issuetype`, {
          headers: {
            'Authorization': `Basic ${auth}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          }
        })
      );
      return response.data;
    } catch (error) {
      console.error('Error fetching Jira issue types:', error.response?.data || error.message);
      throw error;
    }
  }
}
