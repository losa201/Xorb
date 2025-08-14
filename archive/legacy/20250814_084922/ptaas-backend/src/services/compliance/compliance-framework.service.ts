import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { ComplianceFramework } from '../entities/compliance-framework.entity';
import { ComplianceControl } from '../entities/compliance-control.entity';
import { TISAXMapping } from './frameworks/tisax.mapping';
import { KRITISMapping } from './frameworks/kritis.mapping';
import { GDPRMapping } from './frameworks/gdpr.mapping';
import { NIS2Mapping } from './frameworks/nis2.mapping';
import { BSI GrundschutzMapping } from './frameworks/bsi-grundschutz.mapping';

@Injectable()
export class ComplianceFrameworkService {
  constructor(
    @InjectRepository(ComplianceFramework)
    private readonly frameworkRepository: Repository<ComplianceFramework>,
    @InjectRepository(ComplianceControl)
    private readonly controlRepository: Repository<ComplianceControl>
  ) {}

  async initializeFrameworks(): Promise<void> {
    const frameworks = [
      {
        name: 'TISAX',
        description: 'Trusted Information Security Assessment Exchange',
        version: '2023',
        levels: ['Basic', 'Standard', 'Extended'],
        mapping: TISAXMapping
      },
      {
        name: 'KRITIS',
        description: 'Critical Infrastructure Protection',
        version: '2023',
        levels: ['Basic', 'Medium', 'High'],
        mapping: KRITISMapping
      },
      {
        name: 'GDPR',
        description: 'General Data Protection Regulation',
        version: '1.0',
        levels: ['Mandatory'],
        mapping: GDPRMapping
      },
      {
        name: 'NIS2',
        description: 'Network and Information Security Directive 2',
        version: '2023',
        levels: ['Essential', 'Important'],
        mapping: NIS2Mapping
      },
      {
        name: 'BSI Grundschutz',
        description: 'Basic Protection for Information Security',
        version: '2023',
        levels: ['Basic', 'Advanced'],
        mapping: BSI GrundschutzMapping
      }
    ];

    for (const framework of frameworks) {
      let existingFramework = await this.frameworkRepository.findOne({
        where: { name: framework.name }
      });

      if (!existingFramework) {
        existingFramework = this.frameworkRepository.create({
          name: framework.name,
          description: framework.description,
          version: framework.version
        });
        await this.frameworkRepository.save(existingFramework);
      }

      // Clear existing controls
      await this.controlRepository.delete({ framework: { id: existingFramework.id } });

      // Add new controls
      for (const [controlId, controlData] of Object.entries(framework.mapping)) {
        const control = this.controlRepository.create({
          framework: { id: existingFramework.id },
          controlId,
          title: controlData.title,
          description: controlData.description,
          requirements: controlData.requirements,
          implementationGuidance: controlData.implementationGuidance,
          assessmentMethods: controlData.assessmentMethods,
          level: controlData.level
        });
        await this.controlRepository.save(control);
      }
    }
  }

  async getFrameworkByName(name: string): Promise<ComplianceFramework> {
    return this.frameworkRepository.findOne({
      where: { name },
      relations: ['controls']
    });
  }

  async getAllFrameworks(): Promise<ComplianceFramework[]> {
    return this.frameworkRepository.find({
      relations: ['controls']
    });
  }

  async getFrameworkControls(frameworkId: string): Promise<ComplianceControl[]> {
    return this.controlRepository.find({
      where: { framework: { id: frameworkId } }
    });
  }
}
