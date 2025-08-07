#!/usr/bin/env python3
"""
XORB PTaaS German Compliance Framework

Comprehensive German regulatory compliance testing for Penetration Testing as a Service:
- GDPR (Datenschutz-Grundverordnung - DSGVO)
- IT-Sicherheitsgesetz (IT-SiG)
- NIS2-Umsetzungs- und Cybersicherheitsstärkungsgesetz (NIS2UmsuCG)
- BSI-Grundschutz
- Bundesdatenschutzgesetz (BDSG)
- Telekommunikation-Telemedien-Datenschutz-Gesetz (TTDSG)
- Kritis-Verordnungen
- Cyber-Resilience Act (CRA)

Author: XORB Platform Team
Version: 2.1.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"

class SeverityLevel(Enum):
    """Severity levels for compliance violations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    id: str
    regulation: str
    article: str
    title: str
    description: str
    requirements: List[str]
    validation_methods: List[str]
    severity: SeverityLevel
    mandatory: bool
    applicable_sectors: List[str]

@dataclass
class ComplianceTestResult:
    """Result of compliance testing"""
    requirement_id: str
    status: ComplianceStatus
    score: float  # 0.0 to 1.0
    findings: List[str]
    recommendations: List[str]
    evidence: Dict[str, Any]
    tested_at: datetime
    next_review: datetime

class GermanComplianceFramework:
    """German regulatory compliance framework for PTaaS"""
    
    def __init__(self):
        self.requirements = self._initialize_german_requirements()
        self.test_results: Dict[str, ComplianceTestResult] = {}
        
    def _initialize_german_requirements(self) -> Dict[str, ComplianceRequirement]:
        """Initialize all German compliance requirements"""
        requirements = {}
        
        # GDPR/DSGVO Requirements
        gdpr_requirements = self._get_gdpr_requirements()
        requirements.update(gdpr_requirements)
        
        # IT-Sicherheitsgesetz (IT-SiG)
        itsig_requirements = self._get_itsig_requirements()
        requirements.update(itsig_requirements)
        
        # NIS2-Umsetzungs- und Cybersicherheitsstärkungsgesetz
        nis2_requirements = self._get_nis2_german_requirements()
        requirements.update(nis2_requirements)
        
        # BSI-Grundschutz
        bsi_requirements = self._get_bsi_grundschutz_requirements()
        requirements.update(bsi_requirements)
        
        # Bundesdatenschutzgesetz (BDSG)
        bdsg_requirements = self._get_bdsg_requirements()
        requirements.update(bdsg_requirements)
        
        # TTDSG Requirements
        ttdsg_requirements = self._get_ttdsg_requirements()
        requirements.update(ttdsg_requirements)
        
        # KRITIS Requirements
        kritis_requirements = self._get_kritis_requirements()
        requirements.update(kritis_requirements)
        
        # Cyber Resilience Act (CRA)
        cra_requirements = self._get_cra_requirements()
        requirements.update(cra_requirements)
        
        return requirements
    
    def _get_gdpr_requirements(self) -> Dict[str, ComplianceRequirement]:
        """GDPR/DSGVO compliance requirements"""
        return {
            "GDPR_ART_5": ComplianceRequirement(
                id="GDPR_ART_5",
                regulation="DSGVO",
                article="Artikel 5",
                title="Grundsätze für die Verarbeitung personenbezogener Daten",
                description="Rechtmäßigkeit, Verarbeitung nach Treu und Glauben, Transparenz",
                requirements=[
                    "Rechtmäßige, faire und transparente Verarbeitung",
                    "Zweckbindung der Datenverarbeitung",
                    "Datenminimierung",
                    "Richtigkeit der Daten",
                    "Speicherbegrenzung",
                    "Integrität und Vertraulichkeit"
                ],
                validation_methods=[
                    "Datenschutz-Folgenabschätzung prüfen",
                    "Verarbeitungsverzeichnis validieren",
                    "Technische und organisatorische Maßnahmen testen",
                    "Aufbewahrungsfristen überprüfen"
                ],
                severity=SeverityLevel.CRITICAL,
                mandatory=True,
                applicable_sectors=["alle"]
            ),
            
            "GDPR_ART_25": ComplianceRequirement(
                id="GDPR_ART_25",
                regulation="DSGVO",
                article="Artikel 25",
                title="Datenschutz durch Technikgestaltung und datenschutzfreundliche Voreinstellungen",
                description="Privacy by Design und Privacy by Default",
                requirements=[
                    "Datenschutz durch Technikgestaltung (Privacy by Design)",
                    "Datenschutzfreundliche Voreinstellungen (Privacy by Default)",
                    "Pseudonymisierung von personenbezogenen Daten",
                    "Minimierung der Verarbeitung personenbezogener Daten"
                ],
                validation_methods=[
                    "Systemarchitektur auf Privacy by Design prüfen",
                    "Standardeinstellungen validieren",
                    "Pseudonymisierungsverfahren testen",
                    "Datenminimierung in Anwendungen prüfen"
                ],
                severity=SeverityLevel.HIGH,
                mandatory=True,
                applicable_sectors=["alle"]
            ),
            
            "GDPR_ART_32": ComplianceRequirement(
                id="GDPR_ART_32",
                regulation="DSGVO",
                article="Artikel 32",
                title="Sicherheit der Verarbeitung",
                description="Technische und organisatorische Maßnahmen",
                requirements=[
                    "Pseudonymisierung und Verschlüsselung",
                    "Dauerhafte Vertraulichkeit, Integrität, Verfügbarkeit",
                    "Wiederherstellbarkeit bei technischen oder physischen Zwischenfällen",
                    "Regelmäßige Überprüfung der Wirksamkeit"
                ],
                validation_methods=[
                    "Verschlüsselungsverfahren testen",
                    "Backup- und Recovery-Prozesse validieren",
                    "Zugriffskontrollen prüfen",
                    "Incident Response Pläne testen"
                ],
                severity=SeverityLevel.CRITICAL,
                mandatory=True,
                applicable_sectors=["alle"]
            ),
            
            "GDPR_ART_33": ComplianceRequirement(
                id="GDPR_ART_33",
                regulation="DSGVO",
                article="Artikel 33",
                title="Meldung von Verletzungen des Schutzes personenbezogener Daten",
                description="Meldepflicht bei Datenschutzverletzungen",
                requirements=[
                    "Meldung an Aufsichtsbehörde innerhalb von 72 Stunden",
                    "Dokumentation aller Datenschutzverletzungen",
                    "Bewertung des Risikos für betroffene Personen",
                    "Benachrichtigung betroffener Personen bei hohem Risiko"
                ],
                validation_methods=[
                    "Breach Detection Systeme testen",
                    "Meldeprozesse validieren",
                    "Dokumentationsverfahren prüfen",
                    "Benachrichtigungssysteme testen"
                ],
                severity=SeverityLevel.HIGH,
                mandatory=True,
                applicable_sectors=["alle"]
            )
        }
    
    def _get_itsig_requirements(self) -> Dict[str, ComplianceRequirement]:
        """IT-Sicherheitsgesetz (IT-SiG) requirements"""
        return {
            "ITSIG_8A": ComplianceRequirement(
                id="ITSIG_8A",
                regulation="IT-SiG",
                article="§ 8a",
                title="Anforderungen an die IT-Sicherheit von Kritischen Infrastrukturen",
                description="IT-Sicherheitsstandards für KRITIS-Betreiber",
                requirements=[
                    "Angemessene organisatorische und technische Vorkehrungen",
                    "Stand der Technik bei IT-Sicherheitsmaßnahmen",
                    "Störungserkennung und -behandlung",
                    "Meldung erheblicher IT-Sicherheitsvorfälle an BSI"
                ],
                validation_methods=[
                    "IT-Sicherheitskonzept prüfen",
                    "Technische Schutzmaßnahmen validieren",
                    "Incident Response Verfahren testen",
                    "Meldeprozesse an BSI validieren"
                ],
                severity=SeverityLevel.CRITICAL,
                mandatory=True,
                applicable_sectors=["KRITIS"]
            ),
            
            "ITSIG_8B": ComplianceRequirement(
                id="ITSIG_8B",
                regulation="IT-SiG",
                article="§ 8b",
                title="Kontaktstelle und Nachweis",
                description="Benennung einer Kontaktstelle und Nachweispflichten",
                requirements=[
                    "Benennung einer Kontaktstelle zum BSI",
                    "Nachweis der Erfüllung der Anforderungen alle zwei Jahre",
                    "Ansprechpartner für IT-Sicherheitsvorfälle",
                    "Dokumentation der Sicherheitsmaßnahmen"
                ],
                validation_methods=[
                    "Kontaktstellenbenennung prüfen",
                    "Nachweisdokumentation validieren",
                    "Kommunikationsverfahren testen",
                    "Dokumentationsqualität bewerten"
                ],
                severity=SeverityLevel.MEDIUM,
                mandatory=True,
                applicable_sectors=["KRITIS"]
            )
        }
    
    def _get_nis2_german_requirements(self) -> Dict[str, ComplianceRequirement]:
        """NIS2-Umsetzungsgesetz requirements"""
        return {
            "NIS2_CYBER_RISK": ComplianceRequirement(
                id="NIS2_CYBER_RISK",
                regulation="NIS2UmsuCG",
                article="§ 23",
                title="Cybersicherheits-Risikomanagement",
                description="Maßnahmen zum Cybersicherheits-Risikomanagement",
                requirements=[
                    "Strategien für das Cybersicherheits-Risikomanagement",
                    "Bewältigung von Cybersicherheitsvorfällen",
                    "Aufrechterhaltung des Geschäftsbetriebs",
                    "Sicherheit der Lieferkette",
                    "Bewertung der Wirksamkeit von Cybersicherheitsmaßnahmen"
                ],
                validation_methods=[
                    "Risikomanagement-Strategie prüfen",
                    "Incident Response Capabilities testen",
                    "Business Continuity Pläne validieren",
                    "Supply Chain Security bewerten",
                    "Wirksamkeitsmessungen überprüfen"
                ],
                severity=SeverityLevel.CRITICAL,
                mandatory=True,
                applicable_sectors=["wesentliche Einrichtungen", "wichtige Einrichtungen"]
            ),
            
            "NIS2_INCIDENT_REPORT": ComplianceRequirement(
                id="NIS2_INCIDENT_REPORT",
                regulation="NIS2UmsuCG",
                article="§ 32",
                title="Meldung erheblicher Cybersicherheitsvorfälle",
                description="Meldepflichten für Cybersicherheitsvorfälle",
                requirements=[
                    "Frühwarnung innerhalb von 24 Stunden",
                    "Zwischenbericht innerhalb von 72 Stunden",
                    "Abschlussbericht innerhalb eines Monats",
                    "Beurteilung der Auswirkungen auf Dienstleistungen"
                ],
                validation_methods=[
                    "Incident Detection Systeme testen",
                    "Meldeverfahren validieren",
                    "Berichterstattungsprozesse prüfen",
                    "Impact Assessment Verfahren bewerten"
                ],
                severity=SeverityLevel.HIGH,
                mandatory=True,
                applicable_sectors=["wesentliche Einrichtungen", "wichtige Einrichtungen"]
            )
        }
    
    def _get_bsi_grundschutz_requirements(self) -> Dict[str, ComplianceRequirement]:
        """BSI-Grundschutz requirements"""
        return {
            "BSI_ISMS": ComplianceRequirement(
                id="BSI_ISMS",
                regulation="BSI-Grundschutz",
                article="BSI-Standard 200-1",
                title="Managementsysteme für Informationssicherheit (ISMS)",
                description="Aufbau und Betrieb eines ISMS",
                requirements=[
                    "Informationssicherheitsleitlinie etablieren",
                    "Organisationsstrukturen für Informationssicherheit",
                    "Ressourcen für Informationssicherheit bereitstellen",
                    "Integration in Geschäftsprozesse",
                    "Kontinuierliche Verbesserung"
                ],
                validation_methods=[
                    "ISMS-Dokumentation prüfen",
                    "Organisationsstrukturen bewerten",
                    "Ressourcenzuteilung validieren",
                    "Prozessintegration testen",
                    "Verbesserungsprozesse überprüfen"
                ],
                severity=SeverityLevel.HIGH,
                mandatory=True,
                applicable_sectors=["Bundesverwaltung", "KRITIS"]
            ),
            
            "BSI_RISK_ANALYSIS": ComplianceRequirement(
                id="BSI_RISK_ANALYSIS",
                regulation="BSI-Grundschutz",
                article="BSI-Standard 200-3",
                title="Risikoanalyse auf der Basis von IT-Grundschutz",
                description="Durchführung von Risikoanalysen",
                requirements=[
                    "Schutzbedarfsfeststellung durchführen",
                    "Modellierung des Informationsverbunds",
                    "IT-Grundschutz-Check durchführen",
                    "Risikoanalyse für nicht abgedeckte Bereiche",
                    "Behandlung identifizierter Risiken"
                ],
                validation_methods=[
                    "Schutzbedarfsanalyse validieren",
                    "Modellierungsqualität bewerten",
                    "Grundschutz-Check durchführen",
                    "Risikobehandlung prüfen",
                    "Dokumentationsqualität bewerten"
                ],
                severity=SeverityLevel.HIGH,
                mandatory=True,
                applicable_sectors=["Bundesverwaltung", "KRITIS"]
            )
        }
    
    def _get_bdsg_requirements(self) -> Dict[str, ComplianceRequirement]:
        """Bundesdatenschutzgesetz (BDSG) requirements"""
        return {
            "BDSG_22": ComplianceRequirement(
                id="BDSG_22",
                regulation="BDSG",
                article="§ 22",
                title="Verarbeitung zu anderen Zwecken durch öffentliche Stellen",
                description="Zweckänderung bei der Datenverarbeitung",
                requirements=[
                    "Prüfung der Vereinbarkeit mit ursprünglichem Zweck",
                    "Wahrung der Interessen betroffener Personen",
                    "Angemessene Sicherheitsmaßnahmen",
                    "Information der betroffenen Personen"
                ],
                validation_methods=[
                    "Zweckvereinbarkeitsanalyse durchführen",
                    "Interessenabwägung dokumentieren",
                    "Sicherheitsmaßnahmen validieren",
                    "Informationspflichten prüfen"
                ],
                severity=SeverityLevel.MEDIUM,
                mandatory=True,
                applicable_sectors=["öffentliche Stellen"]
            ),
            
            "BDSG_64": ComplianceRequirement(
                id="BDSG_64",
                regulation="BDSG",
                article="§ 64",
                title="Datengeheimnis",
                description="Verpflichtung zur Wahrung des Datengeheimnisses",
                requirements=[
                    "Verpflichtung aller Beschäftigten auf Datengeheimnis",
                    "Fortbestand der Verpflichtung nach Ende der Tätigkeit",
                    "Schulung der Beschäftigten zum Datenschutz",
                    "Dokumentation der Verpflichtungen"
                ],
                validation_methods=[
                    "Verpflichtungserklärungen prüfen",
                    "Schulungsnachweise validieren",
                    "Dokumentationsvollständigkeit bewerten",
                    "Awareness-Programme testen"
                ],
                severity=SeverityLevel.MEDIUM,
                mandatory=True,
                applicable_sectors=["alle"]
            )
        }
    
    def _get_ttdsg_requirements(self) -> Dict[str, ComplianceRequirement]:
        """TTDSG (Telekommunikation-Telemedien-Datenschutz-Gesetz) requirements"""
        return {
            "TTDSG_25": ComplianceRequirement(
                id="TTDSG_25",
                regulation="TTDSG",
                article="§ 25",
                title="Schutz der Endeinrichtungen",
                description="Einwilligung bei Zugriff auf Endeinrichtungen",
                requirements=[
                    "Einwilligung vor Speicherung oder Zugriff auf Endeinrichtungen",
                    "Klare und verständliche Informationen",
                    "Möglichkeit zur Verweigerung der Einwilligung",
                    "Technisch notwendige Speicherung ausgenommen"
                ],
                validation_methods=[
                    "Cookie-Banner und Einwilligungsverfahren testen",
                    "Informationsqualität bewerten",
                    "Opt-out Mechanismen prüfen",
                    "Notwendigkeitsanalyse durchführen"
                ],
                severity=SeverityLevel.HIGH,
                mandatory=True,
                applicable_sectors=["Telemediendiensteanbieter"]
            ),
            
            "TTDSG_26": ComplianceRequirement(
                id="TTDSG_26",
                regulation="TTDSG",
                article="§ 26",
                title="Anerkannte Dienste zur Einwilligungsverwaltung",
                description="Nutzung anerkannter Consent Management Platforms",
                requirements=[
                    "Verwendung anerkannter Einwilligungsverwaltungsdienste",
                    "Technische Standards für Einwilligungssignale",
                    "Interoperabilität zwischen verschiedenen Diensten",
                    "Transparenz über verwendete Dienste"
                ],
                validation_methods=[
                    "CMP-Zertifizierung prüfen",
                    "Technische Standards validieren",
                    "Interoperabilitätstests durchführen",
                    "Transparenzanforderungen bewerten"
                ],
                severity=SeverityLevel.MEDIUM,
                mandatory=False,
                applicable_sectors=["Telemediendiensteanbieter"]
            )
        }
    
    def _get_kritis_requirements(self) -> Dict[str, ComplianceRequirement]:
        """KRITIS-spezifische Anforderungen"""
        return {
            "KRITIS_ENERGY": ComplianceRequirement(
                id="KRITIS_ENERGY",
                regulation="EnWG",
                article="§ 11 Abs. 1a",
                title="IT-Sicherheit in der Energiewirtschaft",
                description="IT-Sicherheitsmaßnahmen für Energieversorger",
                requirements=[
                    "Angemessene IT-Sicherheitsmaßnahmen nach Stand der Technik",
                    "Schutz vor Bedrohungen der IT-Sicherheit",
                    "Kontinuierliche Überwachung der IT-Systeme",
                    "Meldung erheblicher Störungen an Bundesnetzagentur"
                ],
                validation_methods=[
                    "IT-Sicherheitskonzept validieren",
                    "Bedrohungsanalyse durchführen",
                    "Monitoring-Systeme testen",
                    "Meldeprozesse überprüfen"
                ],
                severity=SeverityLevel.CRITICAL,
                mandatory=True,
                applicable_sectors=["Energieversorgung"]
            ),
            
            "KRITIS_WATER": ComplianceRequirement(
                id="KRITIS_WATER",
                regulation="WHG",
                article="§ 21",
                title="Wasserwirtschaftliche Sicherheit",
                description="Sicherheitsmaßnahmen in der Wasserwirtschaft",
                requirements=[
                    "Schutz kritischer Wasserversorgungsanlagen",
                    "Redundante Systeme für kritische Funktionen",
                    "Notfallpläne für Versorgungsunterbrechungen",
                    "Regelmäßige Sicherheitsüberprüfungen"
                ],
                validation_methods=[
                    "Anlagensicherheit bewerten",
                    "Redundanzsysteme testen",
                    "Notfallpläne validieren",
                    "Sicherheitsaudits durchführen"
                ],
                severity=SeverityLevel.CRITICAL,
                mandatory=True,
                applicable_sectors=["Wasserversorgung"]
            )
        }
    
    def _get_cra_requirements(self) -> Dict[str, ComplianceRequirement]:
        """Cyber Resilience Act (CRA) requirements"""
        return {
            "CRA_SECURITY_BY_DESIGN": ComplianceRequirement(
                id="CRA_SECURITY_BY_DESIGN",
                regulation="CRA",
                article="Artikel 10",
                title="Cybersicherheitsanforderungen",
                description="Security by Design für digitale Produkte",
                requirements=[
                    "Sichere Standardkonfiguration",
                    "Schutz vor bekannten Schwachstellen",
                    "Sichere Kommunikation und Authentifizierung",
                    "Logging und Monitoring von Sicherheitsereignissen",
                    "Regelmäßige Sicherheitsupdates"
                ],
                validation_methods=[
                    "Standardkonfiguration prüfen",
                    "Vulnerability Assessment durchführen",
                    "Kommunikationssicherheit testen",
                    "Logging-Mechanismen validieren",
                    "Update-Prozesse bewerten"
                ],
                severity=SeverityLevel.HIGH,
                mandatory=True,
                applicable_sectors=["Hersteller digitaler Produkte"]
            ),
            
            "CRA_VULNERABILITY_HANDLING": ComplianceRequirement(
                id="CRA_VULNERABILITY_HANDLING",
                regulation="CRA",
                article="Artikel 11",
                title="Schwachstellenbehandlung",
                description="Verfahren zur Behandlung von Schwachstellen",
                requirements=[
                    "Koordinierte Schwachstellenoffenlegung",
                    "Risikobewertung von Schwachstellen",
                    "Zeitnahe Bereitstellung von Sicherheitsupdates",
                    "Information der Nutzer über Schwachstellen",
                    "Dokumentation der Schwachstellenbehandlung"
                ],
                validation_methods=[
                    "Disclosure-Prozesse prüfen",
                    "Risikobewertungsverfahren validieren",
                    "Update-Mechanismen testen",
                    "Nutzerkommunikation bewerten",
                    "Dokumentationsqualität prüfen"
                ],
                severity=SeverityLevel.HIGH,
                mandatory=True,
                applicable_sectors=["Hersteller digitaler Produkte"]
            )
        }
    
    async def run_compliance_assessment(
        self, 
        target_system: str,
        applicable_sectors: List[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive German compliance assessment"""
        
        logger.info(f"Starting German compliance assessment for {target_system}")
        
        if applicable_sectors is None:
            applicable_sectors = ["alle"]
        
        assessment_results = {}
        overall_score = 0.0
        total_requirements = 0
        
        for req_id, requirement in self.requirements.items():
            # Check if requirement is applicable
            if not self._is_requirement_applicable(requirement, applicable_sectors):
                continue
            
            logger.info(f"Testing requirement: {req_id} - {requirement.title}")
            
            # Run compliance test
            result = await self._test_requirement(requirement, target_system)
            assessment_results[req_id] = result
            self.test_results[req_id] = result
            
            overall_score += result.score
            total_requirements += 1
        
        final_score = overall_score / total_requirements if total_requirements > 0 else 0.0
        
        # Generate compliance report
        report = self._generate_compliance_report(
            target_system, 
            assessment_results, 
            final_score,
            applicable_sectors
        )
        
        logger.info(f"Compliance assessment completed - Overall score: {final_score:.1%}")
        
        return report
    
    def _is_requirement_applicable(
        self, 
        requirement: ComplianceRequirement,
        applicable_sectors: List[str]
    ) -> bool:
        """Check if requirement is applicable to target sectors"""
        
        if "alle" in requirement.applicable_sectors:
            return True
        
        return any(sector in requirement.applicable_sectors for sector in applicable_sectors)
    
    async def _test_requirement(
        self,
        requirement: ComplianceRequirement,
        target_system: str
    ) -> ComplianceTestResult:
        """Test individual compliance requirement"""
        
        # Simulate compliance testing
        await asyncio.sleep(0.1)  # Simulate testing time
        
        # For demonstration, generate realistic test results
        findings = []
        recommendations = []
        score = 0.8  # Default compliance score
        status = ComplianceStatus.COMPLIANT
        
        # Simulate requirement-specific testing
        if requirement.regulation == "DSGVO":
            findings, recommendations, score, status = await self._test_gdpr_requirement(requirement)
        elif requirement.regulation == "IT-SiG":
            findings, recommendations, score, status = await self._test_itsig_requirement(requirement)
        elif requirement.regulation == "NIS2UmsuCG":
            findings, recommendations, score, status = await self._test_nis2_requirement(requirement)
        elif requirement.regulation == "BSI-Grundschutz":
            findings, recommendations, score, status = await self._test_bsi_requirement(requirement)
        elif requirement.regulation == "BDSG":
            findings, recommendations, score, status = await self._test_bdsg_requirement(requirement)
        elif requirement.regulation == "TTDSG":
            findings, recommendations, score, status = await self._test_ttdsg_requirement(requirement)
        elif requirement.regulation in ["EnWG", "WHG"]:
            findings, recommendations, score, status = await self._test_kritis_requirement(requirement)
        elif requirement.regulation == "CRA":
            findings, recommendations, score, status = await self._test_cra_requirement(requirement)
        
        return ComplianceTestResult(
            requirement_id=requirement.id,
            status=status,
            score=score,
            findings=findings,
            recommendations=recommendations,
            evidence={
                "target_system": target_system,
                "test_methods": requirement.validation_methods,
                "regulation": requirement.regulation,
                "article": requirement.article
            },
            tested_at=datetime.utcnow(),
            next_review=datetime.utcnow() + timedelta(days=90)
        )
    
    async def _test_gdpr_requirement(self, requirement: ComplianceRequirement) -> tuple:
        """Test GDPR/DSGVO specific requirements"""
        findings = []
        recommendations = []
        score = 0.85
        status = ComplianceStatus.COMPLIANT
        
        if requirement.id == "GDPR_ART_5":
            findings = [
                "Rechtmäßigkeitsgrundlagen dokumentiert",
                "Verarbeitungsverzeichnis vorhanden",
                "Zweckbindung eingehalten",
                "Aufbewahrungsfristen definiert"
            ]
            recommendations = [
                "Regelmäßige Überprüfung der Rechtmäßigkeitsgrundlagen",
                "Automatisierte Löschung implementieren",
                "Datenschutz-Folgenabschätzung aktualisieren"
            ]
        elif requirement.id == "GDPR_ART_25":
            findings = [
                "Privacy by Design Prinzipien implementiert",
                "Datenschutzfreundliche Voreinstellungen aktiv",
                "Pseudonymisierung teilweise implementiert"
            ]
            recommendations = [
                "Vollständige Pseudonymisierung aller personenbezogenen Daten",
                "Erweiterte Privacy by Design Maßnahmen",
                "Regelmäßige Privacy Impact Assessments"
            ]
            score = 0.75
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        
        return findings, recommendations, score, status
    
    async def _test_itsig_requirement(self, requirement: ComplianceRequirement) -> tuple:
        """Test IT-Sicherheitsgesetz specific requirements"""
        findings = []
        recommendations = []
        score = 0.9
        status = ComplianceStatus.COMPLIANT
        
        if requirement.id == "ITSIG_8A":
            findings = [
                "IT-Sicherheitskonzept nach Stand der Technik implementiert",
                "Störungserkennungssysteme aktiv",
                "Incident Response Prozesse etabliert",
                "BSI-Meldeverfahren implementiert"
            ]
            recommendations = [
                "Regelmäßige Aktualisierung der Sicherheitsmaßnahmen",
                "Erweiterte Threat Detection Capabilities",
                "Automatisierte BSI-Meldungen implementieren"
            ]
        
        return findings, recommendations, score, status
    
    async def _test_nis2_requirement(self, requirement: ComplianceRequirement) -> tuple:
        """Test NIS2 specific requirements"""
        findings = []
        recommendations = []
        score = 0.8
        status = ComplianceStatus.COMPLIANT
        
        if requirement.id == "NIS2_CYBER_RISK":
            findings = [
                "Cybersicherheits-Risikomanagement-Strategie vorhanden",
                "Incident Response Capabilities implementiert",
                "Business Continuity Pläne etabliert",
                "Supply Chain Security Maßnahmen teilweise implementiert"
            ]
            recommendations = [
                "Vollständige Supply Chain Risk Assessment",
                "Erweiterte Cyber Resilience Maßnahmen",
                "Quantitative Risikobewertung implementieren"
            ]
            score = 0.75
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        
        return findings, recommendations, score, status
    
    async def _test_bsi_requirement(self, requirement: ComplianceRequirement) -> tuple:
        """Test BSI-Grundschutz specific requirements"""
        findings = []
        recommendations = []
        score = 0.85
        status = ComplianceStatus.COMPLIANT
        
        if requirement.id == "BSI_ISMS":
            findings = [
                "ISMS nach BSI-Standard 200-1 implementiert",
                "Informationssicherheitsleitlinie etabliert",
                "Organisationsstrukturen definiert",
                "Kontinuierliche Verbesserungsprozesse aktiv"
            ]
            recommendations = [
                "Erweiterte ISMS-Integration in Geschäftsprozesse",
                "Automatisierte Compliance-Überwachung",
                "Regelmäßige ISMS-Audits durchführen"
            ]
        
        return findings, recommendations, score, status
    
    async def _test_bdsg_requirement(self, requirement: ComplianceRequirement) -> tuple:
        """Test BDSG specific requirements"""
        findings = []
        recommendations = []
        score = 0.9
        status = ComplianceStatus.COMPLIANT
        
        if requirement.id == "BDSG_64":
            findings = [
                "Alle Beschäftigten auf Datengeheimnis verpflichtet",
                "Schulungsprogramme zum Datenschutz implementiert",
                "Verpflichtungserklärungen dokumentiert"
            ]
            recommendations = [
                "Regelmäßige Auffrischungsschulungen",
                "Erweiterte Sensibilisierungsmaßnahmen",
                "Automatisierte Compliance-Überwachung"
            ]
        
        return findings, recommendations, score, status
    
    async def _test_ttdsg_requirement(self, requirement: ComplianceRequirement) -> tuple:
        """Test TTDSG specific requirements"""
        findings = []
        recommendations = []
        score = 0.7
        status = ComplianceStatus.PARTIALLY_COMPLIANT
        
        if requirement.id == "TTDSG_25":
            findings = [
                "Cookie-Banner implementiert",
                "Grundlegende Einwilligungsverfahren vorhanden",
                "Opt-out Mechanismen teilweise implementiert"
            ]
            recommendations = [
                "Vollständige TTDSG-konforme Consent Management Platform",
                "Granulare Einwilligungsoptionen implementieren",
                "Erweiterte Cookie-Kategorisierung",
                "Automatisierte Einwilligungsverfolgung"
            ]
        
        return findings, recommendations, score, status
    
    async def _test_kritis_requirement(self, requirement: ComplianceRequirement) -> tuple:
        """Test KRITIS specific requirements"""
        findings = []
        recommendations = []
        score = 0.85
        status = ComplianceStatus.COMPLIANT
        
        if requirement.id.startswith("KRITIS"):
            findings = [
                "KRITIS-spezifische Sicherheitsmaßnahmen implementiert",
                "Redundante Systeme für kritische Funktionen vorhanden",
                "Notfallpläne etabliert und getestet",
                "Regelmäßige Sicherheitsüberprüfungen durchgeführt"
            ]
            recommendations = [
                "Erweiterte Cyber-Physical Security Maßnahmen",
                "Automatisierte Threat Detection für OT-Systeme",
                "Erweiterte Business Continuity Pläne",
                "Sektor-spezifische Sicherheitsstandards implementieren"
            ]
        
        return findings, recommendations, score, status
    
    async def _test_cra_requirement(self, requirement: ComplianceRequirement) -> tuple:
        """Test Cyber Resilience Act specific requirements"""
        findings = []
        recommendations = []
        score = 0.6
        status = ComplianceStatus.PARTIALLY_COMPLIANT
        
        if requirement.id == "CRA_SECURITY_BY_DESIGN":
            findings = [
                "Grundlegende Security by Design Prinzipien implementiert",
                "Sichere Standardkonfiguration teilweise vorhanden",
                "Update-Mechanismen implementiert"
            ]
            recommendations = [
                "Vollständige CRA-konforme Security by Design Implementation",
                "Automatisierte Vulnerability Scanning",
                "Erweiterte Secure Development Lifecycle",
                "Kontinuierliche Security Monitoring"
            ]
        
        return findings, recommendations, score, status
    
    def _generate_compliance_report(
        self,
        target_system: str,
        results: Dict[str, ComplianceTestResult],
        overall_score: float,
        applicable_sectors: List[str]
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        # Categorize results by regulation
        regulation_scores = {}
        regulation_results = {}
        
        for req_id, result in results.items():
            regulation = self.requirements[req_id].regulation
            
            if regulation not in regulation_scores:
                regulation_scores[regulation] = []
                regulation_results[regulation] = []
            
            regulation_scores[regulation].append(result.score)
            regulation_results[regulation].append(result)
        
        # Calculate regulation-specific scores
        regulation_summary = {}
        for regulation, scores in regulation_scores.items():
            avg_score = sum(scores) / len(scores)
            regulation_summary[regulation] = {
                "average_score": avg_score,
                "total_requirements": len(scores),
                "compliant": len([r for r in regulation_results[regulation] if r.status == ComplianceStatus.COMPLIANT]),
                "partially_compliant": len([r for r in regulation_results[regulation] if r.status == ComplianceStatus.PARTIALLY_COMPLIANT]),
                "non_compliant": len([r for r in regulation_results[regulation] if r.status == ComplianceStatus.NON_COMPLIANT])
            }
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            overall_score, 
            regulation_summary,
            applicable_sectors
        )
        
        # Critical findings and recommendations
        critical_findings = []
        high_priority_recommendations = []
        
        for result in results.values():
            requirement = self.requirements[result.requirement_id]
            
            if result.status == ComplianceStatus.NON_COMPLIANT and requirement.severity == SeverityLevel.CRITICAL:
                critical_findings.append({
                    "requirement_id": result.requirement_id,
                    "regulation": requirement.regulation,
                    "title": requirement.title,
                    "findings": result.findings
                })
            
            if requirement.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                high_priority_recommendations.extend(result.recommendations)
        
        return {
            "assessment_summary": {
                "target_system": target_system,
                "assessment_date": datetime.utcnow().isoformat(),
                "applicable_sectors": applicable_sectors,
                "overall_score": overall_score,
                "compliance_level": self._determine_compliance_level(overall_score),
                "total_requirements_tested": len(results),
                "regulations_covered": list(regulation_summary.keys())
            },
            "executive_summary": executive_summary,
            "regulation_breakdown": regulation_summary,
            "critical_findings": critical_findings,
            "high_priority_recommendations": list(set(high_priority_recommendations)),
            "detailed_results": {req_id: asdict(result) for req_id, result in results.items()},
            "next_steps": self._generate_next_steps(overall_score, critical_findings),
            "compliance_roadmap": self._generate_compliance_roadmap(results, regulation_summary)
        }
    
    def _generate_executive_summary(
        self,
        overall_score: float,
        regulation_summary: Dict[str, Any],
        applicable_sectors: List[str]
    ) -> str:
        """Generate executive summary for compliance report"""
        
        compliance_level = self._determine_compliance_level(overall_score)
        
        summary = f"""
        EXECUTIVE SUMMARY - German Compliance Assessment
        
        Overall Compliance Score: {overall_score:.1%}
        Compliance Level: {compliance_level}
        Applicable Sectors: {', '.join(applicable_sectors)}
        
        Regulation-Specific Performance:
        """
        
        for regulation, data in regulation_summary.items():
            summary += f"\n- {regulation}: {data['average_score']:.1%} ({data['compliant']} compliant, {data['partially_compliant']} partial, {data['non_compliant']} non-compliant)"
        
        if overall_score >= 0.9:
            summary += "\n\nThe assessed system demonstrates excellent compliance with German regulatory requirements."
        elif overall_score >= 0.75:
            summary += "\n\nThe assessed system shows good compliance with minor improvements needed."
        elif overall_score >= 0.6:
            summary += "\n\nThe assessed system requires significant improvements to achieve full compliance."
        else:
            summary += "\n\nThe assessed system has substantial compliance gaps requiring immediate attention."
        
        return summary
    
    def _determine_compliance_level(self, score: float) -> str:
        """Determine compliance level based on score"""
        if score >= 0.95:
            return "Vollständig Compliant"
        elif score >= 0.85:
            return "Weitgehend Compliant"
        elif score >= 0.75:
            return "Überwiegend Compliant"
        elif score >= 0.6:
            return "Teilweise Compliant"
        else:
            return "Nicht Compliant"
    
    def _generate_next_steps(
        self,
        overall_score: float,
        critical_findings: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate next steps based on assessment results"""
        
        next_steps = []
        
        if critical_findings:
            next_steps.append("Sofortige Behebung kritischer Compliance-Verstöße")
            next_steps.append("Implementierung von Notfall-Compliance-Maßnahmen")
        
        if overall_score < 0.75:
            next_steps.append("Entwicklung eines umfassenden Compliance-Verbesserungsplans")
            next_steps.append("Zuweisung dedizierter Compliance-Ressourcen")
        
        next_steps.extend([
            "Regelmäßige Compliance-Überwachung implementieren",
            "Schulung der Mitarbeiter zu deutschen Compliance-Anforderungen",
            "Dokumentation aller Compliance-Maßnahmen aktualisieren",
            "Externe Compliance-Validierung durch Drittanbieter",
            "Kontinuierliche Überwachung regulatorischer Änderungen"
        ])
        
        return next_steps
    
    def _generate_compliance_roadmap(
        self,
        results: Dict[str, ComplianceTestResult],
        regulation_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate compliance improvement roadmap"""
        
        roadmap = {
            "immediate_actions": [],  # 0-30 days
            "short_term_goals": [],   # 1-3 months
            "medium_term_goals": [],  # 3-12 months
            "long_term_strategy": []  # 12+ months
        }
        
        # Immediate actions for critical non-compliance
        for req_id, result in results.items():
            requirement = self.requirements[req_id]
            
            if (result.status == ComplianceStatus.NON_COMPLIANT and 
                requirement.severity == SeverityLevel.CRITICAL):
                roadmap["immediate_actions"].append(f"Behebung: {requirement.title}")
        
        # Short-term goals for high-priority improvements
        for req_id, result in results.items():
            requirement = self.requirements[req_id]
            
            if (result.status == ComplianceStatus.PARTIALLY_COMPLIANT and 
                requirement.severity == SeverityLevel.HIGH):
                roadmap["short_term_goals"].append(f"Verbesserung: {requirement.title}")
        
        # Medium-term goals based on regulation performance
        for regulation, data in regulation_summary.items():
            if data["average_score"] < 0.8:
                roadmap["medium_term_goals"].append(f"Umfassende {regulation} Compliance-Verbesserung")
        
        # Long-term strategy
        roadmap["long_term_strategy"] = [
            "Implementierung eines kontinuierlichen Compliance-Monitoring-Systems",
            "Integration von Compliance in alle Geschäftsprozesse",
            "Aufbau einer Compliance-Kultur im Unternehmen",
            "Proaktive Anpassung an neue regulatorische Anforderungen",
            "Automatisierung von Compliance-Prozessen"
        ]
        
        return roadmap

async def main():
    """Example usage of German Compliance Framework"""
    
    framework = GermanComplianceFramework()
    
    # Run assessment for different sectors
    target_system = "XORB PTaaS Platform"
    
    # Example 1: General IT service provider
    print("🇩🇪 Running German Compliance Assessment - IT Service Provider")
    results_it = await framework.run_compliance_assessment(
        target_system,
        applicable_sectors=["alle", "IT-Dienstleister"]
    )
    
    print(f"Overall Score: {results_it['assessment_summary']['overall_score']:.1%}")
    print(f"Compliance Level: {results_it['assessment_summary']['compliance_level']}")
    
    # Example 2: KRITIS energy sector
    print("\n🏭 Running German Compliance Assessment - KRITIS Energy")
    results_kritis = await framework.run_compliance_assessment(
        target_system,
        applicable_sectors=["KRITIS", "Energieversorgung"]
    )
    
    print(f"Overall Score: {results_kritis['assessment_summary']['overall_score']:.1%}")
    print(f"Compliance Level: {results_kritis['assessment_summary']['compliance_level']}")
    
    # Save results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    with open(f"/root/Xorb/german_compliance_report_{timestamp}.json", 'w') as f:
        json.dump(results_it, f, indent=2, default=str)
    
    print(f"\n📄 German compliance report saved: german_compliance_report_{timestamp}.json")

if __name__ == "__main__":
    asyncio.run(main())