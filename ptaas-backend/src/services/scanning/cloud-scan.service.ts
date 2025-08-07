import { Injectable } from '@nestjs/common';
import { ScanType } from '../scan.enums';
import { xorBScanClient } from '../../clients/xorb-scan.client';
import { ScanRequestDto } from '../dto/scan-request.dto';
import { CloudProvider } from './cloud-provider.enum';

@Injectable()
export class CloudScanService {
  constructor(private readonly scanClient: xorBScanClient) {}

  async scanAwsAccount(scanRequest: ScanRequestDto, awsAccountId: string, regions: string[]) {
    const scanId = await this.scanClient.startCloudScan({
      scanType: ScanType.CLOUD,
      cloudProvider: CloudProvider.AWS,
      accountId: awsAccountId,
      regions,
      scanRequest
    });
    
    return { scanId, provider: CloudProvider.AWS, accountId: awsAccountId };
  }

  async scanAzureSubscription(scanRequest: ScanRequestDto, subscriptionId: string, resourceGroups: string[]) {
    const scanId = await this.scanClient.startCloudScan({
      scanType: ScanType.CLOUD,
      cloudProvider: CloudProvider.AZURE,
      subscriptionId,
      resourceGroups,
      scanRequest
    });
    
    return { scanId, provider: CloudProvider.AZURE, subscriptionId };
  }

  async scanGcpProject(scanRequest: ScanRequestDto, projectId: string, regions: string[]) {
    const scanId = await this.scanClient.startCloudScan({
      scanType: ScanType.CLOUD,
      cloudProvider: CloudProvider.GCP,
      projectId,
      regions,
      scanRequest
    });
    
    return { scanId, provider: CloudProvider.GCP, projectId };
  }

  async getScanResults(scanId: string) {
    return this.scanClient.getCloudScanResults(scanId);
  }
}