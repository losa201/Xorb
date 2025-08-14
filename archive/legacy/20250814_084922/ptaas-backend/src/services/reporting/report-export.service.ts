import { Injectable } from '@nestjs/common';
import { ReportFormat } from '../enums/report-format.enum';
import { PDFReportGenerator } from './generators/pdf-report.generator';
import { HTMLReportGenerator } from './generators/html-report.generator';
import { JSONReportGenerator } from './generators/json-report.generator';
import { CSVReportGenerator } from './generators/csv-report.generator';
import { BrandingOptions } from '../interfaces/branding-options.interface';
import { ReportData } from '../interfaces/report-data.interface';

@Injectable()
export class ReportExportService {
  constructor(
    private readonly pdfGenerator: PDFReportGenerator,
    private readonly htmlGenerator: HTMLReportGenerator,
    private readonly jsonGenerator: JSONReportGenerator,
    private readonly csvGenerator: CSVReportGenerator
  ) {}

  async exportReport(
    reportData: ReportData,
    format: ReportFormat,
    brandingOptions: BrandingOptions
  ): Promise<Buffer> {
    switch (format) {
      case ReportFormat.PDF:
        return this.pdfGenerator.generate(reportData, brandingOptions);
      case ReportFormat.HTML:
        return this.htmlGenerator.generate(reportData, brandingOptions);
      case ReportFormat.JSON:
        return this.jsonGenerator.generate(reportData);
      case ReportFormat.CSV:
        return this.csvGenerator.generate(reportData);
      default:
        throw new Error(`Unsupported report format: ${format}`);
    }
  }
}
