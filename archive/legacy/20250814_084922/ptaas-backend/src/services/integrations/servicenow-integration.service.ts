import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';
import { IntegrationConfig } from '../interfaces/integration-config.interface';
import { ServiceNowConfig } from '../interfaces/servicenow-config.interface';
import { ServiceNowIncident } from '../interfaces/servicenow-incident.interface';
import { ServiceNowIncidentCreateDto } from '../dto/servicenow-incident-create.dto';
import { ServiceNowIncidentUpdateDto } from '../dto/servicenow-incident-update.dto';

@Injectable()
export class ServiceNowIntegrationService {
  private readonly servicenowConfig: ServiceNowConfig;

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService,
  ) {
    this.servicenowConfig = this.configService.get<ServiceNowConfig>('servicenow');
  }

  async createIncident(incidentData: ServiceNowIncidentCreateDto): Promise<ServiceNowIncident> {
    try {
      const auth = Buffer.from(
        `${this.servicenowConfig.username}:${this.servicenowConfig.password}`
      ).toString('base64');

      const headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': `Basic ${auth}`,
      };

      const response = await firstValueFrom(
        this.httpService.post(
          `${this.servicenowConfig.apiUrl}/incident`,
          {
            incident: incidentData,
          },
          { headers }
        )
      );

      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async updateIncident(
    incidentNumber: string,
    updateData: ServiceNowIncidentUpdateDto
  ): Promise<ServiceNowIncident> {
    try {
      const auth = Buffer.from(
        `${this.servicenowConfig.username}:${this.servicenowConfig.password}`
      ).toString('base64');

      const headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': `Basic ${auth}`,
      };

      const response = await firstValueFrom(
        this.httpService.put(
          `${this.servicenowConfig.apiUrl}/incident/${incidentNumber}`,
          {
            incident: updateData,
          },
          { headers }
        )
      );

      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async getIncident(incidentNumber: string): Promise<ServiceNowIncident> {
    try {
      const auth = Buffer.from(
        `${this.servicenowConfig.username}:${this.servicenowConfig.password}`
      ).toString('base64');

      const headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': `Basic ${auth}`,
      };

      const response = await firstValueFrom(
        this.httpService.get(
          `${this.servicenowConfig.apiUrl}/incident/${incidentNumber}`,
          { headers }
        )
      );

      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async searchIncidents(query: string): Promise<ServiceNowIncident[]> {
    try {
      const auth = Buffer.from(
        `${this.servicenowConfig.username}:${this.servicenowConfig.password}`
      ).toString('base64');

      const headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': `Basic ${auth}`,
      };

      const response = await firstValueFrom(
        this.httpService.get(
          `${this.servicenowConfig.apiUrl}/incident?sysparm_query=${query}`,
          { headers }
        )
      );

      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  private handleError(error: any): never {
    // Log error details
    console.error('ServiceNow Integration Error:', {
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      message: error.message,
    });

    // Throw a more user-friendly error
    throw new Error(`ServiceNow integration failed: ${error.message}`);
  }
}
