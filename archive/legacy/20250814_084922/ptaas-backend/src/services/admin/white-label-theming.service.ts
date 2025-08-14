import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join } from 'path';

@Injectable()
export class WhiteLabelThemingService {
  private readonly themesDir: string;

  constructor(private readonly configService: ConfigService) {
    this.themesDir = this.configService.get<string>('WHITE_LABEL_THEMES_DIR', join(process.cwd(), 'themes'));
  }

  /**
   * Get all available white-label themes
   * @returns Array of theme names
   */
  getAvailableThemes(): string[] {
    // Implementation would scan the themes directory
    // This is a simplified example
    return ['default', 'corporate', 'dark', 'light'];
  }

  /**
   * Get configuration for a specific theme
   * @param themeName Name of the theme
   * @returns Theme configuration
   */
  getThemeConfiguration(themeName: string): any {
    const themePath = join(this.themesDir, themeName, 'config.json');

    if (!existsSync(themePath)) {
      throw new Error(`Theme configuration not found for ${themeName}`);
    }

    const themeConfig = readFileSync(themePath, 'utf-8');
    return JSON.parse(themeConfig);
  }

  /**
   * Update theme configuration
   * @param themeName Name of the theme
   * @param config New configuration
   * @returns Success status
   */
  updateThemeConfiguration(themeName: string, config: any): boolean {
    const themePath = join(this.themesDir, themeName, 'config.json');

    if (!existsSync(themePath)) {
      throw new Error(`Theme configuration not found for ${themeName}`);
    }

    writeFileSync(themePath, JSON.stringify(config, null, 2), 'utf-8');
    return true;
  }

  /**
   * Upload custom logo for a theme
   * @param themeName Name of the theme
   * @param logoBuffer Buffer containing the logo image
   * @param mimeType MIME type of the image
   * @returns URL to the uploaded logo
   */
  uploadThemeLogo(themeName: string, logoBuffer: Buffer, mimeType: string): string {
    const allowedMimeTypes = ['image/png', 'image/jpeg', 'image/svg+xml'];

    if (!allowedMimeTypes.includes(mimeType)) {
      throw new Error(`Unsupported logo format: ${mimeType}`);
    }

    const logoDir = join(this.themesDir, themeName, 'assets');
    const logoPath = join(logoDir, 'logo' + this.getFileExtension(mimeType));

    // Implementation would save the logo and return a URL
    // This is a simplified example
    writeFileSync(logoPath, logoBuffer);

    return `/themes/${themeName}/assets/logo${this.getFileExtension(mimeType)}`;
  }

  /**
   * Get file extension for a MIME type
   * @param mimeType MIME type
   * @returns File extension
   */
  private getFileExtension(mimeType: string): string {
    switch (mimeType) {
      case 'image/png': return '.png';
      case 'image/jpeg': return '.jpg';
      case 'image/svg+xml': return '.svg';
      default: return '';
    }
  }
}
