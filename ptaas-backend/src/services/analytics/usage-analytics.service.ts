import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Scan } from '../../entities/scan.entity';
import { User } from '../../entities/user.entity';
import { Client } from '../../entities/client.entity';
import { UsageAnalyticsDto } from '../../dto/analytics/usage-analytics.dto';
import { DateRangeDto } from '../../dto/analytics/date-range.dto';
import { TenantContext } from '../../decorators/tenant.decorator';

@Injectable()
export class UsageAnalyticsService {
  constructor(
    @InjectRepository(Scan)
    private readonly scanRepository: Repository<Scan>,
    @InjectRepository(User)
    private readonly userRepository: Repository<User>,
    @InjectRepository(Client)
    private readonly clientRepository: Repository<Client>
  ) {}

  async getPlatformUsageAnalytics(
    @TenantContext() tenantId: string,
    dateRange: DateRangeDto
  ): Promise<UsageAnalyticsDto> {
    const { startDate, endDate } = dateRange;
    
    // Get total scans
    const totalScans = await this.scanRepository.count({
      where: {
        tenantId,
        createdAt: Between(startDate, endDate)
      }
    });
    
    // Get new users
    const newUsers = await this.userRepository.count({
      where: {
        tenantId,
        createdAt: Between(startDate, endDate)
      }
    });
    
    // Get active clients
    const activeClients = await this.clientRepository.count({
      where: {
        tenantId,
        lastActive: Between(startDate, endDate)
      }
    });
    
    // Get scan type distribution
    const scanTypeDistribution = await this.scanRepository
      .createQueryBuilder('scan')
      .select('scan.type', 'type')
      .addSelect('COUNT(scan.id)', 'count')
      .where('scan.tenantId = :tenantId', { tenantId })
      .andWhere('scan.createdAt BETWEEN :startDate AND :endDate', { startDate, endDate })
      .groupBy('scan.type')
      .getRawMany();
    
    // Get daily scan count
    const dailyScanCount = await this.scanRepository
      .createQueryBuilder('scan')
      .select('DATE(scan.createdAt)', 'date')
      .addSelect('COUNT(scan.id)', 'count')
      .where('scan.tenantId = :tenantId', { tenantId })
      .andWhere('scan.createdAt BETWEEN :startDate AND :endDate', { startDate, endDate })
      .groupBy('DATE(scan.createdAt)')
      .orderBy('DATE(scan.createdAt)', 'ASC')
      .getRawMany();
    
    return {
      totalScans,
      newUsers,
      activeClients,
      scanTypeDistribution,
      dailyScanCount,
      dateRange
    };
  }

  async getClientUsageAnalytics(
    @TenantContext() tenantId: string,
    clientId: string,
    dateRange: DateRangeDto
  ): Promise<UsageAnalyticsDto> {
    const { startDate, endDate } = dateRange;
    
    // Verify client belongs to tenant
    const client = await this.clientRepository.findOne({
      where: {
        id: clientId,
        tenantId
      }
    });
    
    if (!client) {
      throw new Error('Client not found');
    }
    
    // Get client scans
    const clientScans = await this.scanRepository.count({
      where: {
        clientId,
        createdAt: Between(startDate, endDate)
      }
    });
    
    // Get scan type distribution for client
    const scanTypeDistribution = await this.scanRepository
      .createQueryBuilder('scan')
      .select('scan.type', 'type')
      .addSelect('COUNT(scan.id)', 'count')
      .where('scan.clientId = :clientId', { clientId })
      .andWhere('scan.createdAt BETWEEN :startDate AND :endDate', { startDate, endDate })
      .groupBy('scan.type')
      .getRawMany();
    
    // Get daily scan count for client
    const dailyScanCount = await this.scanRepository
      .createQueryBuilder('scan')
      .select('DATE(scan.createdAt)', 'date')
      .addSelect('COUNT(scan.id)', 'count')
      .where('scan.clientId = :clientId', { clientId })
      .andWhere('scan.createdAt BETWEEN :startDate AND :endDate', { startDate, endDate })
      .groupBy('DATE(scan.createdAt)')
      .orderBy('DATE(scan.createdAt)', 'ASC')
      .getRawMany();
    
    return {
      totalScans: clientScans,
      newUsers: 0, // Not applicable for client-level analytics
      activeClients: 1, // The client itself
      scanTypeDistribution,
      dailyScanCount,
      dateRange
    };
  }

  async getScanTrends(
    @TenantContext() tenantId: string,
    dateRange: DateRangeDto
  ): Promise<any> {
    const { startDate, endDate } = dateRange;
    
    // Get weekly scan count for the past year
    const weeklyScanCount = await this.scanRepository
      .createQueryBuilder('scan')
      .select('DATE_TRUNC('week', scan.createdAt)', 'week')
      .addSelect('COUNT(scan.id)', 'count')
      .where('scan.tenantId = :tenantId', { tenantId })
      .andWhere('scan.createdAt BETWEEN :startDate AND :endDate', { 
        startDate: new Date(new Date().setFullYear(new Date().getFullYear() - 1)),
        endDate: new Date()
      })
      .groupBy('DATE_TRUNC('week', scan.createdAt)')
      .orderBy('DATE_TRUNC('week', scan.createdAt)', 'ASC')
      .getRawMany();
    
    // Get scan type trends
    const scanTypeTrends = await this.scanRepository
      .createQueryBuilder('scan')
      .select('DATE_TRUNC('month', scan.createdAt)', 'month')
      .addSelect('scan.type', 'type')
      .addSelect('COUNT(scan.id)', 'count')
      .where('scan.tenantId = :tenantId', { tenantId })
      .andWhere('scan.createdAt BETWEEN :startDate AND :endDate', { 
        startDate: new Date(new Date().setFullYear(new Date().getFullYear() - 1)),
        endDate: new Date()
      })
      .groupBy('DATE_TRUNC('month', scan.createdAt)', 'scan.type')
      .orderBy('DATE_TRUNC('month', scan.createdAt)', 'ASC')
      .addOrderBy('scan.type', 'ASC')
      .getRawMany();
    
    return {
      weeklyScanCount,
      scanTypeTrends
    };
  }
}