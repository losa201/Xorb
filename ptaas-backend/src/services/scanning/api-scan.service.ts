import { Injectable } from '@nestjs/common';
import { ScanType, ScanRequest, ScanResult } from '../scan.interface';
import { XORBClient } from '../../clients/xorb.client';

@Injectable()
export class APIScanService {
  constructor(private readonly xorbClient: XORBClient) {}

  async scanREST(request: ScanRequest): Promise<ScanResult> {
    return this.xorbClient.startScan({
      ...request,
      scanType: ScanType.REST_API
    });
  }

  async scanGraphQL(request: ScanRequest): Promise<ScanResult> {
    return this.xorbClient.startScan({
      ...request,
      scanType: ScanType.GRAPHQL
    });
  }

  async scangRPC(request: ScanRequest): Promise<ScanResult> {
    return this.xorbClient.startScan({
      ...request,
      scanType: ScanType.GRPC
    });
  }
}