import { Injectable } from '@nestjs/common';
import { XorbAIClient } from '../../clients/xorb-ai.client';
import { Vulnerability } from '../../entities/vulnerability.entity';
import { Language } from '../../enums/language.enum';

@Injectable()
export class AIRemediationService {
  constructor(private readonly xorbAIClient: XorbAIClient) {}

  async generateRemediationGuidance(
    vulnerability: Vulnerability,
    language: Language = Language.EN
  ): Promise<string> {
    const prompt = this.buildRemediationPrompt(vulnerability, language);

    try {
      const response = await this.xorbAIClient.generateText({
        prompt,
        model: 'xorb-ai-2.1',
        temperature: 0.3,
        maxTokens: 1000
      });

      return response.text;
    } catch (error) {
      // Log error and return fallback guidance
      console.error(`AI remediation generation failed: ${error.message}`);
      return this.getFallbackRemediation(vulnerability.type, language);
    }
  }

  private buildRemediationPrompt(vulnerability: Vulnerability, language: Language): string {
    return `Generate detailed remediation guidance for the following vulnerability in ${language === Language.DE ? 'German' : 'English'}:

Type: ${vulnerability.type}
Severity: ${vulnerability.severity}
Description: ${vulnerability.description}

Include:
1. Immediate mitigation steps
2. Long-term remediation strategy
3. Code examples (if applicable)
4. Compliance considerations
5. Verification steps

Follow these guidelines:
- Use clear, technical language appropriate for security teams
- Prioritize practical, actionable advice
- Include references to relevant security standards (e.g., OWASP, NIST)
- Format the response in markdown with clear sections`;
  }

  private getFallbackRemediation(type: string, language: Language): string {
    // In a real implementation, this would use i18n and a database of common remediations
    if (language === Language.DE) {
      return `Standard-Reparaturanleitung für ${type} (vorübergehend)

1. Zugriff auf das betroffene System beschränken
2. Sicherstellen, dass alle Systeme auf dem neuesten Patch-Stand sind
3. Überwachung für verdächtige Aktivitäten einrichten
4. Kontextspezifische Maßnahmen für ${type} implementieren

Diese Anleitung ist vorübergehend und sollte durch eine detailliertere Analyse ersetzt werden.`;
    }

    return `Standard remediation for ${type} (temporary)

1. Restrict access to the affected system
2. Ensure all systems are up to date with security patches
3. Set up monitoring for suspicious activity
4. Implement context-specific measures for ${type}

This guidance is temporary and should be replaced with more detailed analysis.`;
  }
}
