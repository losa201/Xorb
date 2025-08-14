import { Injectable } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { ScanService } from './scan.service';
import { ContinuousScanConfig, ScanType } from '../types/scan.types';
import { XORBAIClient } from '../clients/xorb-ai.client';

@Injectable()
export class ScanSchedulerService {
  private continuousScans: Map<string, ContinuousScanConfig> = new Map();

  constructor(
    private readonly scanService: ScanService,
    private readonly xorbAIClient: XORBAIClient
  ) {}

  @Cron(CronExpression.EVERY_HOUR)
  async handleHourlyScans() {
    await this.executeContinuousScans('hourly');
  }

  @Cron(CronExpression.EVERY_6_HOURS)
  async handleSixHourlyScans() {
    await this.executeContinuousScans('six_hourly');
  }

  @Cron(CronExpression.EVERY_DAY_AT_MIDNIGHT)
  async handleDailyScans() {
    await this.executeContinuousScans('daily');
  }

  async addContinuousScan(config: ContinuousScanConfig): Promise<string> {
    const scanId = `continuous_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.continuousScans.set(scanId, config);
    return scanId;
  }

  async removeContinuousScan(scanId: string): Promise<boolean> {
    return this.continuousScans.delete(scanId);
  }

  async getContinuousScan(scanId: string): Promise<ContinuousScanConfig | undefined> {
    return this.continuousScans.get(scanId);
  }

  async listContinuousScans(): Promise<ContinuousScanConfig[]> {
    return Array.from(this.continuousScans.values());
  }

  private async executeContinuousScans(frequency: string): Promise<void> {
    const now = new Date();

    for (const [scanId, config] of this.continuousScans.entries()) {
      if (config.frequency === frequency && this.isTimeToScan(config, now)) {
        try {
          // Get AI recommendations for scan configuration
          const aiRecommendations = await this.xorbAIClient.getScanRecommendations({
            asset: config.asset,
            scanType: config.scanType,
            lastScanDate: config.lastScanDate
          });

          // Update scan config with AI recommendations
          const updatedConfig = {
            ...config,
            scanConfig: {
              ...config.scanConfig,
              ...aiRecommendations.scanConfig
            }
          };

          // Execute the scan
          const scanResult = await this.scanService.executeScan(
            updatedConfig.asset,
            updatedConfig.scanType,
            updatedConfig.scanConfig
          );

          // Update last scan date
          updatedConfig.lastScanDate = now;
          this.continuousScans.set(scanId, updatedConfig);

          // Handle scan result (store, notify, etc.)
          this.handleScanResult(scanId, scanResult);
        } catch (error) {
          this.handleScanError(scanId, error);
        }
      }
    }
  }

  private isTimeToScan(config: ContinuousScanConfig, now: Date): boolean {
    if (!config.lastScanDate) {
      return true; // First time scan
    }

    const lastScan = new Date(config.lastScanDate);
    const timeDiff = now.getTime() - lastScan.getTime();

    switch (config.frequency) {
      case 'hourly':
        return timeDiff >= 60 * 60 * 1000;
      case 'six_hourly':
        return timeDiff >= 6 * 60 * 60 * 1000;
      case 'daily':
        return timeDiff >= 24 * 60 * 60 * 1000;
      default:
        return false;
    }
  }

  private handleScanResult(scanId: string, result: any): void {
    // Store result, send notifications, etc.
    // This could emit events to be consumed by other services
    console.log(`Scan completed successfully: ${scanId}`);
  }

  private handleScanError(scanId: string, error: Error): void {
    // Log error, send alerts, etc.
    console.error(`Scan failed: ${scanId}`, error);
  }
}

// Continuous scan configuration interface
export interface ContinuousScanConfig {
  asset: string; // Asset ID or URL to scan
  scanType: ScanType; // Type of scan to perform
  scanConfig: any; // Configuration for the scan
  frequency: 'hourly' | 'six_hourly' | 'daily'; // How often to run the scan
  lastScanDate?: Date; // When the scan was last executed
  enabled?: boolean; // Whether the scan is currently active
  ownerId: string; // ID of the user/organization that owns this scan
  name: string; // Name of the continuous scan
  description?: string; // Description of the scan
  notifications?: {
    email?: string[];
    webhook?: string;
    slack?: string;
  };
  retentionPeriod?: number; // How long to keep scan results (in days)
  maxConcurrentScans?: number; // Maximum number of concurrent scans allowed
}
