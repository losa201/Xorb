import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Client } from '../../entities/client.entity';
import { User } from '../../entities/user.entity';
import { AuditLogService } from '../audit/audit-log.service';

@Injectable()
export class ClientSwitchingService {
  constructor(
    @InjectRepository(Client)
    private readonly clientRepository: Repository<Client>,
    @InjectRepository(User)
    private readonly userRepository: Repository<User>,
    private readonly auditLogService: AuditLogService
  ) {}

  async switchClient(userId: string, clientId: string): Promise<{ success: boolean; message: string; newClient: Client }> {
    // Verify user exists
    const user = await this.userRepository.findOne({
      where: { id: userId },
      relations: ['clients', 'currentClient']
    });

    if (!user) {
      throw new Error('User not found');
    }

    // Verify client exists and user has access
    const client = await this.clientRepository.findOne({
      where: { id: clientId }
    });

    if (!client || !user.clients.some(c => c.id === clientId)) {
      throw new Error('User does not have access to this client');
    }

    // Update user's current client
    user.currentClient = client;
    await this.userRepository.save(user);

    // Log the client switch
    await this.auditLogService.logClientSwitch(userId, clientId);

    return {
      success: true,
      message: `Successfully switched to client ${client.name}`,
      newClient: client
    };
  }

  async getClientUsers(clientId: string): Promise<User[]> {
    return this.userRepository.find({
      where: { clients: { id: clientId } },
      select: ['id', 'email', 'firstName', 'lastName', 'role']
    });
  }

  async getClientDetails(clientId: string): Promise<Client> {
    const client = await this.clientRepository.findOne({
      where: { id: clientId },
      relations: ['users', 'assets', 'scans', 'reports']
    });

    if (!client) {
      throw new Error('Client not found');
    }

    return client;
  }

  async listAllClients(): Promise<Client[]> {
    return this.clientRepository.find({
      select: ['id', 'name', 'organization', 'createdAt']
    });
  }
}
