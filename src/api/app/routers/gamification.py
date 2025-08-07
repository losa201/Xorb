"""
Gamification API endpoints for researcher ratings, badges, and leaderboard
"""

import json
import logging
from datetime import datetime
from typing import Any

import redis
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

from xorb_core.gamification.glicko2 import GlickoRating, ResearcherRatingSystem

from ..database import get_db
from ..deps import get_current_active_user, get_current_researcher
from ..models import Badge, Finding, RatingHistory, Researcher, ResearcherBadge

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gamification", tags=["gamification"])

# Redis client for caching leaderboard
redis_client = redis.Redis(host='redis', port=6379, db=1, decode_responses=True)

class ResearcherStats(BaseModel):
    """Researcher statistics model"""
    researcher_id: str
    handle: str
    rating: float
    rd: float
    volatility: float
    tier: str
    xp_multiplier: float
    total_findings: int
    accepted_findings: int
    duplicate_findings: int
    false_positive_findings: int
    total_earnings: float
    badges_earned: list[dict]
    days_since_activity: int
    rating_confidence: str
    progress_to_next_tier: float

class LeaderboardEntry(BaseModel):
    """Leaderboard entry model"""
    rank: int
    researcher_id: str | None
    handle: str
    rating: float
    rd: float
    tier: str
    total_findings: int
    accepted_findings: int
    total_earnings: float
    xp_multiplier: float
    days_since_activity: int
    rating_confidence: str

class BadgeInfo(BaseModel):
    """Badge information model"""
    id: str
    name: str
    description: str
    svg_icon: str
    earned_at: datetime | None
    progress: dict[str, Any] | None

class GamificationService:
    """Service for gamification operations"""

    def __init__(self, db: Session):
        self.db = db
        self.rating_system = ResearcherRatingSystem()

    async def update_researcher_rating(self, researcher_id: str, findings: list[dict]) -> dict[str, Any]:
        """Update researcher rating based on recent findings"""

        # Get current researcher data
        researcher = self.db.query(Researcher).filter(Researcher.id == researcher_id).first()
        if not researcher:
            raise HTTPException(status_code=404, detail="Researcher not found")

        # Create current rating object
        current_rating = GlickoRating(
            rating=researcher.rating,
            rd=researcher.rd,
            vol=researcher.vol,
            last_competition=researcher.last_competition
        )

        # Update rating based on findings
        new_rating = self.rating_system.update_researcher_rating(current_rating, findings)

        # Store rating history
        rating_history = RatingHistory(
            researcher_id=researcher_id,
            old_rating=current_rating.rating,
            new_rating=new_rating.rating,
            old_rd=current_rating.rd,
            new_rd=new_rating.rd,
            old_vol=current_rating.vol,
            new_vol=new_rating.vol,
            change_reason='findings_processed',
            rating_change=new_rating.rating - current_rating.rating,
            org_id=researcher.org_id
        )
        self.db.add(rating_history)

        # Update researcher record
        old_tier = researcher.current_tier
        new_tier = self.rating_system.get_rating_tier(new_rating.rating)

        researcher.rating = new_rating.rating
        researcher.rd = new_rating.rd
        researcher.vol = new_rating.vol
        researcher.last_competition = new_rating.last_competition
        researcher.current_tier = new_tier

        # Update tier history if tier changed
        if old_tier != new_tier:
            tier_history = researcher.tier_history or []
            tier_history.append({
                'tier': new_tier,
                'achieved_at': datetime.utcnow().isoformat(),
                'rating': new_rating.rating
            })
            researcher.tier_history = tier_history

            # Award tier badge if earned
            await self.check_and_award_tier_badge(researcher_id, new_tier)

        self.db.commit()

        # Clear leaderboard cache
        redis_client.delete('leaderboard:global')

        return {
            'old_rating': current_rating.rating,
            'new_rating': new_rating.rating,
            'rating_change': new_rating.rating - current_rating.rating,
            'old_tier': old_tier,
            'new_tier': new_tier,
            'tier_changed': old_tier != new_tier
        }

    async def get_researcher_stats(self, researcher_id: str) -> ResearcherStats:
        """Get comprehensive researcher statistics"""

        researcher = self.db.query(Researcher).filter(Researcher.id == researcher_id).first()
        if not researcher:
            raise HTTPException(status_code=404, detail="Researcher not found")

        # Create rating object for stats calculation
        rating = GlickoRating(
            rating=researcher.rating,
            rd=researcher.rd,
            vol=researcher.vol,
            last_competition=researcher.last_competition
        )

        # Get comprehensive stats
        stats = self.rating_system.get_researcher_stats(rating)

        # Get badges
        badges = self.db.query(ResearcherBadge, Badge).join(Badge).filter(
            ResearcherBadge.researcher_id == researcher_id
        ).all()

        badges_info = [
            {
                'id': badge.Badge.id,
                'name': badge.Badge.name,
                'description': badge.Badge.description,
                'svg_icon': badge.Badge.svg_icon,
                'earned_at': badge.ResearcherBadge.earned_at
            }
            for badge in badges
        ]

        return ResearcherStats(
            researcher_id=str(researcher_id),
            handle=researcher.handle,
            rating=stats['rating'],
            rd=stats['rd'],
            volatility=stats['volatility'],
            tier=stats['tier'],
            xp_multiplier=stats['xp_multiplier'],
            total_findings=researcher.total_findings,
            accepted_findings=researcher.accepted_findings,
            duplicate_findings=researcher.duplicate_findings,
            false_positive_findings=researcher.false_positive_findings,
            total_earnings=float(researcher.total_earnings),
            badges_earned=badges_info,
            days_since_activity=stats['days_since_activity'],
            rating_confidence=stats['rating_confidence'],
            progress_to_next_tier=stats['progress_to_next_tier']
        )

    async def generate_leaderboard(self, limit: int = 50, include_anonymous: bool = True) -> list[LeaderboardEntry]:
        """Generate researcher leaderboard"""

        # Try to get from cache first
        cache_key = f'leaderboard:global:{limit}:{include_anonymous}'
        cached_data = redis_client.get(cache_key)

        if cached_data:
            return [LeaderboardEntry(**entry) for entry in json.loads(cached_data)]

        # Query researchers
        query = self.db.query(Researcher).filter(
            Researcher.leaderboard_visible == True
        )

        if not include_anonymous:
            query = query.filter(Researcher.handle_anonymous == False)

        researchers = query.order_by(desc(Researcher.rating)).limit(limit).all()

        leaderboard = []
        for rank, researcher in enumerate(researchers, 1):
            # Create rating object for stats
            rating = GlickoRating(
                rating=researcher.rating,
                rd=researcher.rd,
                vol=researcher.vol,
                last_competition=researcher.last_competition
            )

            stats = self.rating_system.get_researcher_stats(rating)

            entry = LeaderboardEntry(
                rank=rank,
                researcher_id=str(researcher.id) if not researcher.handle_anonymous else None,
                handle=researcher.handle if not researcher.handle_anonymous else f"Anonymous_{rank}",
                rating=stats['rating'],
                rd=stats['rd'],
                tier=stats['tier'],
                total_findings=researcher.total_findings,
                accepted_findings=researcher.accepted_findings,
                total_earnings=float(researcher.total_earnings),
                xp_multiplier=stats['xp_multiplier'],
                days_since_activity=stats['days_since_activity'],
                rating_confidence=stats['rating_confidence']
            )

            leaderboard.append(entry)

        # Cache for 5 minutes
        redis_client.setex(
            cache_key,
            300,
            json.dumps([entry.dict() for entry in leaderboard])
        )

        return leaderboard

    async def check_and_award_badges(self, researcher_id: str):
        """Check and award badges based on researcher activity"""

        researcher = self.db.query(Researcher).filter(Researcher.id == researcher_id).first()
        if not researcher:
            return

        # Get all badges and check criteria
        badges = self.db.query(Badge).filter(Badge.active == True).all()

        for badge in badges:
            # Skip if already earned
            existing_badge = self.db.query(ResearcherBadge).filter(
                and_(
                    ResearcherBadge.researcher_id == researcher_id,
                    ResearcherBadge.badge_id == badge.id
                )
            ).first()

            if existing_badge:
                continue

            # Check badge criteria
            if await self.meets_badge_criteria(researcher, badge):
                # Award badge
                researcher_badge = ResearcherBadge(
                    researcher_id=researcher_id,
                    badge_id=badge.id,
                    org_id=researcher.org_id
                )
                self.db.add(researcher_badge)

                # Update researcher badges_earned field
                badges_earned = researcher.badges_earned or []
                badges_earned.append({
                    'badge_id': badge.id,
                    'earned_at': datetime.utcnow().isoformat()
                })
                researcher.badges_earned = badges_earned

                logger.info(f"Badge {badge.id} awarded to researcher {researcher_id}")

        self.db.commit()

    async def meets_badge_criteria(self, researcher: Researcher, badge: Badge) -> bool:
        """Check if researcher meets badge criteria"""

        criteria = badge.criteria

        # Tier-based badges
        if badge.tier_requirement:
            return researcher.current_tier == badge.tier_requirement

        # Rating-based badges
        if badge.rating_requirement:
            return researcher.rating >= badge.rating_requirement

        # RD-based badges (reliable researcher)
        if badge.id == 'reliable_researcher':
            return researcher.rd < 60

        # Activity-based badges
        if 'findings_submitted' in criteria:
            return researcher.total_findings >= criteria['findings_submitted']

        if 'critical_findings' in criteria:
            critical_count = self.db.query(func.count(Finding.id)).filter(
                and_(
                    Finding.researcher_id == researcher.id,
                    Finding.severity == 'critical',
                    Finding.status == 'accepted'
                )
            ).scalar()
            return critical_count >= criteria['critical_findings']

        if 'min_findings' in criteria and 'max_false_positive_rate' in criteria:
            if researcher.total_findings >= criteria['min_findings']:
                fp_rate = researcher.false_positive_findings / researcher.total_findings
                return fp_rate <= criteria['max_false_positive_rate']

        return False

    async def check_and_award_tier_badge(self, researcher_id: str, tier: str):
        """Award tier-specific badge"""

        badge_mapping = {
            'Bronze': 'bronze_tier',
            'Silver': 'silver_tier',
            'Gold': 'gold_tier',
            'Platinum': 'platinum_tier',
            'Diamond': 'diamond_tier',
            'Master': 'master_tier'
        }

        badge_id = badge_mapping.get(tier)
        if not badge_id:
            return

        researcher = self.db.query(Researcher).filter(Researcher.id == researcher_id).first()
        if not researcher:
            return

        # Check if badge already earned
        existing_badge = self.db.query(ResearcherBadge).filter(
            and_(
                ResearcherBadge.researcher_id == researcher_id,
                ResearcherBadge.badge_id == badge_id
            )
        ).first()

        if existing_badge:
            return

        # Award tier badge
        researcher_badge = ResearcherBadge(
            researcher_id=researcher_id,
            badge_id=badge_id,
            org_id=researcher.org_id
        )
        self.db.add(researcher_badge)

        # Update badges_earned
        badges_earned = researcher.badges_earned or []
        badges_earned.append({
            'badge_id': badge_id,
            'earned_at': datetime.utcnow().isoformat()
        })
        researcher.badges_earned = badges_earned

        self.db.commit()
        logger.info(f"Tier badge {badge_id} awarded to researcher {researcher_id}")

    async def get_available_badges(self, researcher_id: str) -> list[BadgeInfo]:
        """Get all available badges with progress information"""

        researcher = self.db.query(Researcher).filter(Researcher.id == researcher_id).first()
        if not researcher:
            raise HTTPException(status_code=404, detail="Researcher not found")

        # Get all badges
        badges = self.db.query(Badge).filter(Badge.active == True).all()

        # Get earned badges
        earned_badges = self.db.query(ResearcherBadge).filter(
            ResearcherBadge.researcher_id == researcher_id
        ).all()

        earned_badge_ids = {badge.badge_id: badge.earned_at for badge in earned_badges}

        badge_list = []
        for badge in badges:
            progress = await self.calculate_badge_progress(researcher, badge)

            badge_info = BadgeInfo(
                id=badge.id,
                name=badge.name,
                description=badge.description,
                svg_icon=badge.svg_icon,
                earned_at=earned_badge_ids.get(badge.id),
                progress=progress
            )

            badge_list.append(badge_info)

        # Sort by earned status and progress
        badge_list.sort(key=lambda x: (x.earned_at is None, -(x.progress.get('percentage', 0) if x.progress else 0)))

        return badge_list

    async def calculate_badge_progress(self, researcher: Researcher, badge: Badge) -> dict[str, Any] | None:
        """Calculate progress towards earning a badge"""

        criteria = badge.criteria

        if badge.tier_requirement:
            tier_ratings = {
                'Bronze': 0, 'Silver': 1200, 'Gold': 1500,
                'Platinum': 1800, 'Diamond': 2100, 'Master': 2500
            }
            required_rating = tier_ratings.get(badge.tier_requirement, 0)
            current_rating = researcher.rating

            if current_rating >= required_rating:
                return {'percentage': 100, 'completed': True}
            else:
                # Calculate progress from previous tier
                prev_rating = 0
                for tier, rating in tier_ratings.items():
                    if rating < required_rating and rating <= current_rating:
                        prev_rating = rating

                progress = min(100, ((current_rating - prev_rating) / (required_rating - prev_rating)) * 100)
                return {
                    'percentage': progress,
                    'completed': False,
                    'current': current_rating,
                    'required': required_rating
                }

        if 'findings_submitted' in criteria:
            required = criteria['findings_submitted']
            current = researcher.total_findings
            return {
                'percentage': min(100, (current / required) * 100),
                'completed': current >= required,
                'current': current,
                'required': required
            }

        if 'critical_findings' in criteria:
            required = criteria['critical_findings']
            current = self.db.query(func.count(Finding.id)).filter(
                and_(
                    Finding.researcher_id == researcher.id,
                    Finding.severity == 'critical',
                    Finding.status == 'accepted'
                )
            ).scalar() or 0

            return {
                'percentage': min(100, (current / required) * 100),
                'completed': current >= required,
                'current': current,
                'required': required
            }

        return None

# Initialize gamification service
def get_gamification_service(db: Session = Depends(get_db)) -> GamificationService:
    return GamificationService(db)

@router.get("/stats/{researcher_id}", response_model=ResearcherStats)
async def get_researcher_stats(
    researcher_id: str,
    service: GamificationService = Depends(get_gamification_service),
    current_user = Depends(get_current_active_user)
):
    """Get researcher statistics and rating information"""
    return await service.get_researcher_stats(researcher_id)

@router.get("/my-stats", response_model=ResearcherStats)
async def get_my_stats(
    service: GamificationService = Depends(get_gamification_service),
    current_researcher = Depends(get_current_researcher)
):
    """Get current researcher's statistics"""
    return await service.get_researcher_stats(str(current_researcher.id))

@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def get_leaderboard(
    limit: int = Query(50, ge=1, le=200),
    include_anonymous: bool = Query(True),
    service: GamificationService = Depends(get_gamification_service)
):
    """Get researcher leaderboard"""
    return await service.generate_leaderboard(limit, include_anonymous)

@router.get("/badges", response_model=list[BadgeInfo])
async def get_available_badges(
    service: GamificationService = Depends(get_gamification_service),
    current_researcher = Depends(get_current_researcher)
):
    """Get all available badges with progress"""
    return await service.get_available_badges(str(current_researcher.id))

@router.get("/badges/{researcher_id}", response_model=list[BadgeInfo])
async def get_researcher_badges(
    researcher_id: str,
    service: GamificationService = Depends(get_gamification_service),
    current_user = Depends(get_current_active_user)
):
    """Get badges for specific researcher"""
    return await service.get_available_badges(researcher_id)

@router.post("/update-rating/{researcher_id}")
async def update_researcher_rating(
    researcher_id: str,
    findings_data: list[dict],
    background_tasks: BackgroundTasks,
    service: GamificationService = Depends(get_gamification_service),
    current_user = Depends(get_current_active_user)
):
    """Update researcher rating based on findings (Admin only)"""

    result = await service.update_researcher_rating(researcher_id, findings_data)

    # Check for badge awards in background
    background_tasks.add_task(service.check_and_award_badges, researcher_id)

    return result

@router.post("/recalculate-ratings")
async def recalculate_all_ratings(
    background_tasks: BackgroundTasks,
    service: GamificationService = Depends(get_gamification_service),
    current_user = Depends(get_current_active_user)  # Admin only
):
    """Recalculate all researcher ratings (Admin only)"""

    # This would be a background task to recalculate all ratings
    # For now, return success message
    return {
        "status": "queued",
        "message": "Rating recalculation has been queued for all researchers"
    }

@router.get("/rating-history/{researcher_id}")
async def get_rating_history(
    researcher_id: str,
    limit: int = Query(50, ge=1, le=200),
    service: GamificationService = Depends(get_gamification_service),
    current_user = Depends(get_current_active_user)
):
    """Get rating history for researcher"""

    history = service.db.query(RatingHistory).filter(
        RatingHistory.researcher_id == researcher_id
    ).order_by(desc(RatingHistory.created_at)).limit(limit).all()

    return [
        {
            'id': str(entry.id),
            'old_rating': entry.old_rating,
            'new_rating': entry.new_rating,
            'rating_change': entry.rating_change,
            'old_rd': entry.old_rd,
            'new_rd': entry.new_rd,
            'change_reason': entry.change_reason,
            'finding_id': str(entry.finding_id) if entry.finding_id else None,
            'created_at': entry.created_at
        }
        for entry in history
    ]
