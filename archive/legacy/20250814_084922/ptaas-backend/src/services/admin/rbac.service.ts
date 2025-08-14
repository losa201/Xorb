import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { User, UserDocument } from '../users/schemas/user.schema';
import { Role, RoleDocument } from './schemas/role.schema';
import { Permission, PermissionDocument } from './schemas/permission.schema';
import { Client, ClientDocument } from './schemas/client.schema';
import {
  CreateRoleDto,
  UpdateRoleDto,
  AssignRoleDto,
  CreateClientDto,
  UpdateClientDto
} from './dto/admin.dto';

@Injectable()
export class RbacService {
  constructor(
    @InjectModel(User.name) private userModel: Model<UserDocument>,
    @InjectModel(Role.name) private roleModel: Model<RoleDocument>,
    @InjectModel(Permission.name) private permissionModel: Model<PermissionDocument>,
    @InjectModel(Client.name) private clientModel: Model<ClientDocument>
  ) {}

  // Client management
  async createClient(createClientDto: CreateClientDto): Promise<Client> {
    const createdClient = new this.clientModel(createClientDto);
    return createdClient.save();
  }

  async getAllClients(): Promise<Client[]> {
    return this.clientModel.find().exec();
  }

  async getClientById(id: string): Promise<Client> {
    return this.clientModel.findById(id).exec();
  }

  async updateClient(id: string, updateClientDto: UpdateClientDto): Promise<Client> {
    return this.clientModel.findByIdAndUpdate(id, updateClientDto, { new: true }).exec();
  }

  async deleteClient(id: string): Promise<boolean> {
    const result = await this.clientModel.findByIdAndDelete(id).exec();
    return !!result;
  }

  // Role management
  async createRole(createRoleDto: CreateRoleDto): Promise<Role> {
    const createdRole = new this.roleModel(createRoleDto);
    return createdRole.save();
  }

  async getAllRoles(): Promise<Role[]> {
    return this.roleModel.find().exec();
  }

  async getRoleById(id: string): Promise<Role> {
    return this.roleModel.findById(id).exec();
  }

  async updateRole(id: string, updateRoleDto: UpdateRoleDto): Promise<Role> {
    return this.roleModel.findByIdAndUpdate(id, updateRoleDto, { new: true }).exec();
  }

  async deleteRole(id: string): Promise<boolean> {
    const result = await this.roleModel.findByIdAndDelete(id).exec();
    return !!result;
  }

  // Permission management
  async createPermission(name: string, description: string): Promise<Permission> {
    const createdPermission = new this.permissionModel({ name, description });
    return createdPermission.save();
  }

  async assignPermissionToRole(roleId: string, permissionId: string): Promise<Role> {
    const role = await this.roleModel.findById(roleId).exec();
    const permission = await this.permissionModel.findById(permissionId).exec();

    if (!role || !permission) {
      throw new Error('Role or permission not found');
    }

    if (!role.permissions.includes(permissionId)) {
      role.permissions.push(permissionId);
      return role.save();
    }

    return role;
  }

  async removePermissionFromRole(roleId: string, permissionId: string): Promise<Role> {
    return this.roleModel.findByIdAndUpdate(
      roleId,
      { $pull: { permissions: permissionId } },
      { new: true }
    ).exec();
  }

  // User role management
  async assignRoleToUser(userId: string, roleId: string): Promise<User> {
    const user = await this.userModel.findById(userId).exec();
    const role = await this.roleModel.findById(roleId).exec();

    if (!user || !role) {
      throw new Error('User or role not found');
    }

    if (!user.roles.includes(roleId)) {
      user.roles.push(roleId);
      return user.save();
    }

    return user;
  }

  async removeRoleFromUser(userId: string, roleId: string): Promise<User> {
    return this.userModel.findByIdAndUpdate(
      userId,
      { $pull: { roles: roleId } },
      { new: true }
    ).exec();
  }

  // Permission checking
  async hasPermission(userId: string, permissionName: string): Promise<boolean> {
    const user = await this.userModel.findById(userId)
      .populate('roles')
      .exec();

    if (!user) {
      return false;
    }

    // If user has admin role, grant all permissions
    const isAdmin = user.roles.some(role =>
      role instanceof this.roleModel && role.name === 'admin'
    );

    if (isAdmin) {
      return true;
    }

    // Check if any of the user's roles have the required permission
    const permission = await this.permissionModel.findOne({ name: permissionName }).exec();

    if (!permission) {
      return false;
    }

    return user.roles.some(roleId =>
      permission.roles.includes(roleId.toString())
    );
  }
}
