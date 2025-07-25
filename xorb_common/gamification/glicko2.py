"""
Glicko-2 Rating System Implementation
For researcher skill assessment and gamification in Xorb PTaaS
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Glicko-2 system constants  
TAU = 0.5  # System constant that controls volatility changes
EPSILON = 0.000001  # Convergence threshold

@dataclass
class GlickoRating:
    """Glicko-2 rating data structure"""
    rating: float = 1500.0      # μ (mu) - rating
    rd: float = 350.0           # φ (phi) - rating deviation  
    vol: float = 0.06           # σ (sigma) - volatility
    last_competition: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_competition is None:
            self.last_competition = datetime.utcnow()

@dataclass 
class GameResult:
    """Individual game/finding result for rating calculation"""
    opponent_rating: float
    opponent_rd: float  
    score: float  # 1.0 = win, 0.5 = draw, 0.0 = loss

class Glicko2Calculator:
    """Glicko-2 rating system calculator"""
    
    def __init__(self, tau: float = TAU):
        self.tau = tau
        
    def scale_down(self, rating: float, rd: float) -> Tuple[float, float]:
        """Convert from Glicko to Glicko-2 scale"""
        mu = (rating - 1500) / 173.7178
        phi = rd / 173.7178
        return mu, phi
        
    def scale_up(self, mu: float, phi: float) -> Tuple[float, float]:
        """Convert from Glicko-2 to Glicko scale"""
        rating = mu * 173.7178 + 1500
        rd = phi * 173.7178
        return rating, rd
        
    def g(self, phi: float) -> float:
        """g(φ) function for Glicko-2"""
        return 1 / math.sqrt(1 + 3 * phi * phi / (math.pi * math.pi))
        
    def E(self, mu: float, mu_j: float, phi_j: float) -> float:
        """E(μ, μⱼ, φⱼ) expected score function"""
        return 1 / (1 + math.exp(-self.g(phi_j) * (mu - mu_j)))
        
    def update_rating(self, player: GlickoRating, results: List[GameResult]) -> GlickoRating:
        """
        Update player rating based on competition results
        
        Args:
            player: Current player rating
            results: List of game results
            
        Returns:
            Updated GlickoRating
        """
        if not results:
            # No games played - apply time-based RD increase
            return self.apply_time_decay(player)
            
        # Convert to Glicko-2 scale
        mu, phi = self.scale_down(player.rating, player.rd)
        sigma = player.vol
        
        # Step 2: Update RD for time passage (if applicable)
        phi = self.apply_time_based_rd_increase(phi, player.last_competition)
        
        # Step 3: Compute the quantity v
        v = self.compute_v(mu, results)
        
        # Step 4: Compute the quantity Δ (Delta)
        delta = self.compute_delta(mu, v, results)
        
        # Step 5: Determine new volatility σ'
        new_sigma = self.compute_new_volatility(sigma, phi, delta, v)
        
        # Step 6: Update rating deviation to φ*
        phi_star = math.sqrt(phi * phi + new_sigma * new_sigma)
        
        # Step 7: Update rating deviation to φ'
        new_phi = 1 / math.sqrt(1 / (phi_star * phi_star) + 1 / v)
        
        # Step 8: Update rating to μ'
        new_mu = mu + new_phi * new_phi * self.compute_rating_change_sum(mu, results)
        
        # Convert back to Glicko scale
        new_rating, new_rd = self.scale_up(new_mu, new_phi)
        
        return GlickoRating(
            rating=max(100, min(3000, new_rating)),  # Clamp rating to reasonable bounds
            rd=max(30, min(350, new_rd)),            # Clamp RD to reasonable bounds  
            vol=max(0.01, min(0.2, new_sigma)),      # Clamp volatility to reasonable bounds
            last_competition=datetime.utcnow()
        )
        
    def compute_v(self, mu: float, results: List[GameResult]) -> float:
        """Compute the quantity v (estimated variance)"""
        v_sum = 0.0
        for result in results:
            mu_j, phi_j = self.scale_down(result.opponent_rating, result.opponent_rd)
            g_phi_j = self.g(phi_j)
            E_val = self.E(mu, mu_j, phi_j)
            v_sum += g_phi_j * g_phi_j * E_val * (1 - E_val)
            
        return 1 / v_sum if v_sum > 0 else float('inf')
        
    def compute_delta(self, mu: float, v: float, results: List[GameResult]) -> float:
        """Compute the quantity Δ (delta)"""
        delta_sum = 0.0
        for result in results:
            mu_j, phi_j = self.scale_down(result.opponent_rating, result.opponent_rd)
            g_phi_j = self.g(phi_j)
            E_val = self.E(mu, mu_j, phi_j)
            delta_sum += g_phi_j * (result.score - E_val)
            
        return v * delta_sum
        
    def compute_rating_change_sum(self, mu: float, results: List[GameResult]) -> float:
        """Compute sum for rating change calculation"""
        rating_sum = 0.0
        for result in results:
            mu_j, phi_j = self.scale_down(result.opponent_rating, result.opponent_rd)
            g_phi_j = self.g(phi_j)
            E_val = self.E(mu, mu_j, phi_j)
            rating_sum += g_phi_j * (result.score - E_val)
            
        return rating_sum
        
    def compute_new_volatility(self, sigma: float, phi: float, delta: float, v: float) -> float:
        """Compute new volatility using Illinois algorithm"""
        # Step 5.1
        a = math.log(sigma * sigma)
        
        # Step 5.2  
        def f(x: float) -> float:
            ex = math.exp(x)
            return (ex * (delta * delta - phi * phi - v - ex)) / \
                   (2 * (phi * phi + v + ex) * (phi * phi + v + ex)) - \
                   (x - a) / (self.tau * self.tau)
        
        # Step 5.3
        A = a
        if delta * delta > phi * phi + v:
            B = math.log(delta * delta - phi * phi - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau
            
        # Step 5.4
        fa = f(A)
        fb = f(B)
        
        # Step 5.5 - Illinois algorithm
        for _ in range(100):  # Maximum iterations to prevent infinite loops
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            
            if abs(fc) < EPSILON:
                break
                
            if fc * fb < 0:
                A = B
                fa = fb
            else:
                fa /= 2
                
            B = C
            fb = fc
            
        # Step 5.6
        return math.exp(C / 2)
        
    def apply_time_based_rd_increase(self, phi: float, last_competition: datetime) -> float:
        """Apply time-based rating deviation increase"""
        if last_competition is None:
            return phi
            
        days_since = (datetime.utcnow() - last_competition).days
        
        # Increase RD by approximately 1 point per day of inactivity
        time_factor = days_since * 0.01
        new_phi = math.sqrt(phi * phi + time_factor * time_factor)
        
        return min(new_phi, 350 / 173.7178)  # Cap at maximum RD
        
    def apply_time_decay(self, player: GlickoRating) -> GlickoRating:
        """Apply time-based rating decay for inactive players"""
        if player.last_competition is None:
            return player
            
        days_since = (datetime.utcnow() - player.last_competition).days
        
        # Increase RD for inactive players
        time_rd_increase = min(days_since * 1.0, 350 - player.rd)
        new_rd = min(player.rd + time_rd_increase, 350)
        
        return GlickoRating(
            rating=player.rating,
            rd=new_rd,
            vol=player.vol,
            last_competition=player.last_competition
        )

class ResearcherRatingSystem:
    """Researcher rating system for Xorb PTaaS gamification"""
    
    def __init__(self):
        self.calculator = Glicko2Calculator()
        
        # Rating tiers for badges
        self.rating_tiers = {
            'Bronze': (0, 1200),
            'Silver': (1200, 1500), 
            'Gold': (1500, 1800),
            'Platinum': (1800, 2100),
            'Diamond': (2100, 2500),
            'Master': (2500, float('inf'))
        }
        
    def calculate_finding_score(self, finding_severity: str, is_duplicate: bool, 
                               is_false_positive: bool) -> float:
        """
        Calculate score for a finding submission
        
        Args:
            finding_severity: Severity level (info, low, medium, high, critical)
            is_duplicate: Whether finding is a duplicate
            is_false_positive: Whether finding is a false positive
            
        Returns:
            Score between 0.0 and 1.0
        """
        if is_false_positive or is_duplicate:
            return 0.0
            
        severity_scores = {
            'info': 0.3,
            'low': 0.4, 
            'medium': 0.6,
            'high': 0.8,
            'critical': 1.0
        }
        
        return severity_scores.get(finding_severity.lower(), 0.3)
        
    def create_game_result(self, finding_severity: str, is_duplicate: bool,
                          is_false_positive: bool, average_researcher_rating: float = 1500.0,
                          average_researcher_rd: float = 200.0) -> GameResult:
        """Create GameResult from finding data"""
        score = self.calculate_finding_score(finding_severity, is_duplicate, is_false_positive)
        
        return GameResult(
            opponent_rating=average_researcher_rating,
            opponent_rd=average_researcher_rd,
            score=score
        )
        
    def update_researcher_rating(self, current_rating: GlickoRating, 
                               findings: List[Dict]) -> GlickoRating:
        """Update researcher rating based on findings"""
        results = []
        
        for finding in findings:
            result = self.create_game_result(
                finding_severity=finding.get('severity', 'info'),
                is_duplicate=finding.get('is_duplicate', False),
                is_false_positive=finding.get('is_false_positive', False)
            )
            results.append(result)
            
        return self.calculator.update_rating(current_rating, results)
        
    def get_rating_tier(self, rating: float) -> str:
        """Get rating tier/badge for given rating"""
        for tier, (min_rating, max_rating) in self.rating_tiers.items():
            if min_rating <= rating < max_rating:
                return tier
        return 'Bronze'
        
    def get_xp_multiplier(self, rd: float) -> float:
        """
        Calculate XP multiplier based on rating deviation
        
        Lower RD = more established rating = higher multiplier
        """
        if rd < 40:
            return 1.20  # +20% for very stable rating
        elif rd < 60:
            return 1.10  # +10% for stable rating
        else:
            return 1.00  # No bonus for uncertain rating
            
    def calculate_payout_bonus(self, base_payout: float, rating: float, rd: float) -> float:
        """Calculate total payout with rating-based bonuses"""
        xp_multiplier = self.get_xp_multiplier(rd)
        return base_payout * xp_multiplier
        
    def get_researcher_stats(self, rating: GlickoRating) -> Dict:
        """Get comprehensive researcher statistics"""
        tier = self.get_rating_tier(rating.rating)
        xp_multiplier = self.get_xp_multiplier(rating.rd)
        
        # Calculate next tier progress
        next_tier_rating = None
        for tier_name, (min_rating, max_rating) in self.rating_tiers.items():
            if min_rating > rating.rating:
                next_tier_rating = min_rating
                break
                
        progress_to_next = 0.0
        if next_tier_rating:
            current_tier_min = None
            for tier_name, (min_rating, max_rating) in self.rating_tiers.items():
                if min_rating <= rating.rating < max_rating:
                    current_tier_min = min_rating
                    break
                    
            if current_tier_min is not None:
                progress_to_next = (rating.rating - current_tier_min) / (next_tier_rating - current_tier_min)
        
        return {
            'rating': round(rating.rating, 1),
            'rd': round(rating.rd, 1), 
            'volatility': round(rating.vol, 3),
            'tier': tier,
            'xp_multiplier': xp_multiplier,
            'days_since_activity': (datetime.utcnow() - rating.last_competition).days if rating.last_competition else 0,
            'next_tier_rating': next_tier_rating,
            'progress_to_next_tier': min(1.0, max(0.0, progress_to_next)),
            'rating_confidence': self.get_rating_confidence(rating.rd)
        }
        
    def get_rating_confidence(self, rd: float) -> str:
        """Get human-readable rating confidence"""
        if rd < 50:
            return 'Very High'
        elif rd < 100:
            return 'High'
        elif rd < 150:
            return 'Medium'
        elif rd < 200:
            return 'Low'
        else:
            return 'Very Low'
            
    def generate_leaderboard(self, researchers: List[Dict]) -> List[Dict]:
        """Generate leaderboard with ratings and stats"""
        leaderboard = []
        
        for researcher in researchers:
            rating_data = researcher.get('rating', {})
            rating = GlickoRating(
                rating=rating_data.get('rating', 1500),
                rd=rating_data.get('rd', 350),
                vol=rating_data.get('vol', 0.06),
                last_competition=datetime.fromisoformat(rating_data['last_competition']) if rating_data.get('last_competition') else None
            )
            
            stats = self.get_researcher_stats(rating)
            
            leaderboard.append({
                'researcher_id': researcher.get('id'),
                'handle': researcher.get('handle'),
                'organization': researcher.get('organization'),
                'rating': stats['rating'],
                'rd': stats['rd'],
                'tier': stats['tier'],
                'xp_multiplier': stats['xp_multiplier'],
                'total_findings': researcher.get('total_findings', 0),
                'accepted_findings': researcher.get('accepted_findings', 0),
                'total_earnings': researcher.get('total_earnings', 0.0),
                'days_since_activity': stats['days_since_activity'],
                'rating_confidence': stats['rating_confidence']
            })
            
        # Sort by rating (descending)
        leaderboard.sort(key=lambda x: x['rating'], reverse=True)
        
        # Add rank
        for i, entry in enumerate(leaderboard):
            entry['rank'] = i + 1
            
        return leaderboard

# Example usage and testing
if __name__ == "__main__":
    # Create rating system
    rating_system = ResearcherRatingSystem()
    
    # Test researcher with initial rating
    researcher_rating = GlickoRating()
    
    # Simulate some findings
    findings = [
        {'severity': 'high', 'is_duplicate': False, 'is_false_positive': False},
        {'severity': 'medium', 'is_duplicate': False, 'is_false_positive': False},
        {'severity': 'low', 'is_duplicate': True, 'is_false_positive': False},
        {'severity': 'critical', 'is_duplicate': False, 'is_false_positive': False},
    ]
    
    # Update rating
    new_rating = rating_system.update_researcher_rating(researcher_rating, findings)
    
    # Get stats
    stats = rating_system.get_researcher_stats(new_rating)
    
    print("Updated Rating Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    print(f"\nRating change: {new_rating.rating - researcher_rating.rating:.1f}")
    print(f"RD change: {new_rating.rd - researcher_rating.rd:.1f}")
    print(f"New tier: {stats['tier']}")
    print(f"XP Multiplier: {stats['xp_multiplier']:.2f}")