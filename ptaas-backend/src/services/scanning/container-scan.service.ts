import { ScanType } from '../../types/scanning';
import { XORBClient } from '../xorb-client';

export class ContainerScanService {
  private xorClient: XORBClient;

  constructor(xorClient: XORBClient) {
    this.xorClient = xorClient;
  }

  async scanDockerImage(imageName: string, scanOptions: ContainerScanOptions): Promise<ScanResult> {
    // Implement Docker image scanning logic using XORB API
    const scanRequest: ContainerScanRequest = {
      type: ScanType.CONTAINER,
      image: imageName,
      options: {
        vulnerabilityCheck: scanOptions.vulnerabilityCheck || true,
        misconfigurationCheck: scanOptions.misconfigurationCheck || true,
        secretsCheck: scanOptions.secretsCheck || true,
        baseImageCheck: scanOptions.baseImageCheck || true
      }
    };

    return this.xorClient.submitScan(scanRequest);
  }

  async scanKubernetesManifest(manifestPath: string, scanOptions: KubernetesScanOptions): Promise<ScanResult> {
    // Implement Kubernetes manifest scanning logic using XORB API
    const scanRequest: KubernetesScanRequest = {
      type: ScanType.KUBERNETES,
      manifestPath: manifestPath,
      options: {
        securityPolicies: scanOptions.securityPolicies || true,
        networkPolicies: scanOptions.networkPolicies || true,
        secretsManagement: scanOptions.secretsManagement || true,
        podSecurity: scanOptions.podSecurity || true
      }
    };

    return this.xorClient.submitScan(scanRequest);
  }

  async getScanStatus(scanId: string): Promise<ScanStatus> {
    return this.xorClient.getScanStatus(scanId);
  }
}

export interface ContainerScanOptions {
  vulnerabilityCheck?: boolean;
  misconfigurationCheck?: boolean;
  secretsCheck?: boolean;
  baseImageCheck?: boolean;
}

export interface KubernetesScanOptions {
  securityPolicies?: boolean;
  networkPolicies?: boolean;
  secretsManagement?: boolean;
  podSecurity?: boolean;
}

export interface ContainerScanRequest {
  type: ScanType;
  image: string;
  options: {
    vulnerabilityCheck: boolean;
    misconfigurationCheck: boolean;
    secretsCheck: boolean;
    baseImageCheck: boolean;
  };
}

export interface KubernetesScanRequest {
  type: ScanType;
  manifestPath: string;
  options: {
    securityPolicies: boolean;
    networkPolicies: boolean;
    secretsManagement: boolean;
    podSecurity: boolean;
  };
}

export interface ScanResult {
  scanId: string;
  status: string;
  findings?: ScanFinding[];
  summary?: ScanSummary;
}

export interface ScanStatus {
  scanId: string;
  status: string;
  progress: number;
  startTime?: Date;
  endTime?: Date;
}

export interface ScanFinding {
  id: string;
  severity: string;
  title: string;
  description: string;
  remediation: string;
  references: string[];
}

export interface ScanSummary {
  totalFindings: number;
  severityCounts: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  scanTime: number;
}