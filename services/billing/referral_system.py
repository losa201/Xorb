"""
Xorb Referral System
Handles referral tracking, rewards, and billing adjustments
"""

import asyncio
import json
import logging
import secrets
import string
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

import asyncpg
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ReferralCode(BaseModel):
    """Referral code model"""
    code: str
    organization_id: str
    reward_type: str  # percentage_discount, fixed_credit, free_months
    reward_value: Decimal
    expires_at: Optional[datetime] = None
    max_uses: Optional[int] = None
    current_uses: int = 0

class ReferralReward(BaseModel):
    """Referral reward model"""
    referrer_organization_id: str
    referred_organization_id: str
    reward_amount: Decimal
    reward_type: str
    applied_at: datetime

class ReferralSystem:
    """Manages referral codes, tracking, and rewards"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    async def initialize_tables(self):
        """Create referral system tables"""
        async with self.db_pool.acquire() as conn:
            # Referral codes table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS referral_codes (
                    id SERIAL PRIMARY KEY,
                    code VARCHAR(20) UNIQUE NOT NULL,
                    organization_id UUID NOT NULL,
                    reward_type VARCHAR(50) NOT NULL,
                    reward_value DECIMAL(10,2) NOT NULL,
                    expires_at TIMESTAMP,
                    max_uses INTEGER,
                    current_uses INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Referral tracking table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS referrals (
                    id SERIAL PRIMARY KEY,
                    referral_code VARCHAR(20) NOT NULL,
                    referrer_organization_id UUID NOT NULL,
                    referred_organization_id UUID NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW(),
                    converted_at TIMESTAMP,
                    FOREIGN KEY (referral_code) REFERENCES referral_codes(code)
                )
            """)
            
            # Referral rewards table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS referral_rewards (
                    id SERIAL PRIMARY KEY,
                    referral_id INTEGER NOT NULL,
                    referrer_organization_id UUID NOT NULL,
                    referred_organization_id UUID NOT NULL,
                    reward_amount DECIMAL(10,2) NOT NULL,
                    reward_type VARCHAR(50) NOT NULL,
                    applied_at TIMESTAMP DEFAULT NOW(),
                    billing_credit_applied BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (referral_id) REFERENCES referrals(id)
                )
            """)
            
            # Referral stats table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS referral_stats (
                    organization_id UUID PRIMARY KEY,
                    total_referrals INTEGER DEFAULT 0,
                    successful_referrals INTEGER DEFAULT 0,
                    total_rewards_earned DECIMAL(10,2) DEFAULT 0.00,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indices
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_referral_codes_org ON referral_codes(organization_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_referrals_code ON referrals(referral_code)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_referrals_referrer ON referrals(referrer_organization_id)")
    
    def generate_referral_code(self, organization_id: str) -> str:
        """Generate unique referral code"""
        # Create a readable code: XORB + 6 random characters
        random_part = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        return f"XORB{random_part}"
    
    async def create_referral_code(
        self,
        organization_id: str,
        reward_type: str = "percentage_discount",
        reward_value: Decimal = Decimal("10.00"),
        expires_at: Optional[datetime] = None,
        max_uses: Optional[int] = None
    ) -> str:
        """Create new referral code for organization"""
        
        # Generate unique code
        attempts = 0
        while attempts < 10:
            code = self.generate_referral_code(organization_id)
            
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO referral_codes (
                            code, organization_id, reward_type, reward_value,
                            expires_at, max_uses
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    """, code, organization_id, reward_type, reward_value,
                         expires_at, max_uses)
                
                logger.info(f"Created referral code {code} for org {organization_id}")
                return code
                
            except asyncpg.UniqueViolationError:
                attempts += 1
                continue
        
        raise ValueError("Failed to generate unique referral code")
    
    async def get_referral_code_info(self, code: str) -> Optional[Dict]:
        """Get referral code information"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM referral_codes WHERE code = $1
            """, code)
            
            if row:
                return dict(row)
            return None
    
    async def validate_referral_code(self, code: str) -> tuple[bool, str]:
        """Validate if referral code can be used"""
        code_info = await self.get_referral_code_info(code)
        
        if not code_info:
            return False, "Invalid referral code"
        
        # Check expiration
        if code_info["expires_at"] and datetime.now() > code_info["expires_at"]:
            return False, "Referral code has expired"
        
        # Check usage limits
        if code_info["max_uses"] and code_info["current_uses"] >= code_info["max_uses"]:
            return False, "Referral code usage limit reached"
        
        return True, "Valid"
    
    async def apply_referral_code(
        self,
        code: str,
        referred_organization_id: str
    ) -> Dict:
        """Apply referral code for new organization"""
        
        # Validate code
        is_valid, message = await self.validate_referral_code(code)
        if not is_valid:
            raise ValueError(message)
        
        code_info = await self.get_referral_code_info(code)
        referrer_org_id = code_info["organization_id"]
        
        # Prevent self-referral
        if referrer_org_id == referred_organization_id:
            raise ValueError("Cannot use your own referral code")
        
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Create referral record
                referral_id = await conn.fetchval("""
                    INSERT INTO referrals (
                        referral_code, referrer_organization_id, referred_organization_id
                    ) VALUES ($1, $2, $3)
                    RETURNING id
                """, code, referrer_org_id, referred_organization_id)
                
                # Update referral code usage
                await conn.execute("""
                    UPDATE referral_codes 
                    SET current_uses = current_uses + 1, updated_at = NOW()
                    WHERE code = $1
                """, code)
                
                # Update referrer stats
                await conn.execute("""
                    INSERT INTO referral_stats (organization_id, total_referrals)
                    VALUES ($1, 1)
                    ON CONFLICT (organization_id) DO UPDATE SET
                        total_referrals = referral_stats.total_referrals + 1,
                        updated_at = NOW()
                """, referrer_org_id)
        
        logger.info(f"Applied referral code {code} for org {referred_organization_id}")
        
        return {
            "referral_id": referral_id,
            "referrer_organization_id": referrer_org_id,
            "reward_type": code_info["reward_type"],
            "reward_value": float(code_info["reward_value"])
        }
    
    async def process_referral_conversion(self, referred_organization_id: str):
        """Process referral conversion when referred org subscribes"""
        
        async with self.db_pool.acquire() as conn:
            # Find pending referral for this organization
            referral = await conn.fetchrow("""
                SELECT r.*, rc.reward_type, rc.reward_value
                FROM referrals r
                JOIN referral_codes rc ON r.referral_code = rc.code
                WHERE r.referred_organization_id = $1 AND r.status = 'pending'
            """, referred_organization_id)
            
            if not referral:
                return None
            
            async with conn.transaction():
                # Mark referral as converted
                await conn.execute("""
                    UPDATE referrals 
                    SET status = 'converted', converted_at = NOW()
                    WHERE id = $1
                """, referral["id"])
                
                # Calculate rewards
                reward_amount = await self.calculate_referral_reward(
                    referral["reward_type"],
                    referral["reward_value"],
                    referred_organization_id
                )
                
                # Create reward record
                reward_id = await conn.fetchval("""
                    INSERT INTO referral_rewards (
                        referral_id, referrer_organization_id, referred_organization_id,
                        reward_amount, reward_type
                    ) VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                """, referral["id"], referral["referrer_organization_id"],
                     referred_organization_id, reward_amount, referral["reward_type"])
                
                # Update referrer stats
                await conn.execute("""
                    UPDATE referral_stats 
                    SET successful_referrals = successful_referrals + 1,
                        total_rewards_earned = total_rewards_earned + $2,
                        updated_at = NOW()
                    WHERE organization_id = $1
                """, referral["referrer_organization_id"], reward_amount)
        
        logger.info(f"Processed referral conversion for org {referred_organization_id}, "
                   f"reward: ${reward_amount}")
        
        return {
            "referral_id": referral["id"],
            "reward_id": reward_id,
            "reward_amount": float(reward_amount),
            "referrer_organization_id": referral["referrer_organization_id"]
        }
    
    async def calculate_referral_reward(
        self,
        reward_type: str,
        reward_value: Decimal,
        referred_organization_id: str
    ) -> Decimal:
        """Calculate actual reward amount based on type"""
        
        if reward_type == "fixed_credit":
            return reward_value
        
        elif reward_type == "percentage_discount":
            # For percentage discounts, we give the referrer a fixed amount
            # based on the referred organization's tier
            # This is a simplified implementation
            return Decimal("25.00")  # $25 credit for referral
        
        elif reward_type == "free_months":
            # Calculate monetary value of free months
            # This would depend on the referred organization's subscription
            return reward_value * Decimal("99.00")  # Assume Growth tier value
        
        else:
            return Decimal("10.00")  # Default reward
    
    async def get_organization_referrals(self, organization_id: str) -> Dict:
        """Get referral information for organization"""
        async with self.db_pool.acquire() as conn:
            # Get referral codes created by this organization
            codes = await conn.fetch("""
                SELECT * FROM referral_codes 
                WHERE organization_id = $1
                ORDER BY created_at DESC
            """, organization_id)
            
            # Get referrals made using these codes
            referrals = await conn.fetch("""
                SELECT r.*, rc.code, rc.reward_type, rc.reward_value
                FROM referrals r
                JOIN referral_codes rc ON r.referral_code = rc.code
                WHERE r.referrer_organization_id = $1
                ORDER BY r.created_at DESC
            """, organization_id)
            
            # Get rewards earned
            rewards = await conn.fetch("""
                SELECT * FROM referral_rewards
                WHERE referrer_organization_id = $1
                ORDER BY applied_at DESC
            """, organization_id)
            
            # Get stats
            stats = await conn.fetchrow("""
                SELECT * FROM referral_stats
                WHERE organization_id = $1
            """, organization_id)
        
        return {
            "organization_id": organization_id,
            "referral_codes": [dict(code) for code in codes],
            "referrals": [dict(referral) for referral in referrals],
            "rewards": [dict(reward) for reward in rewards],
            "stats": dict(stats) if stats else {
                "total_referrals": 0,
                "successful_referrals": 0,
                "total_rewards_earned": 0.00
            }
        }
    
    async def apply_referral_credits(self, organization_id: str) -> Decimal:
        """Apply pending referral credits to organization's billing"""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Get unapplied referral rewards
                rewards = await conn.fetch("""
                    SELECT * FROM referral_rewards
                    WHERE referrer_organization_id = $1 
                    AND billing_credit_applied = FALSE
                """, organization_id)
                
                total_credits = Decimal("0.00")
                reward_ids = []
                
                for reward in rewards:
                    total_credits += reward["reward_amount"]
                    reward_ids.append(reward["id"])
                
                if total_credits > 0:
                    # Mark rewards as applied
                    await conn.execute("""
                        UPDATE referral_rewards 
                        SET billing_credit_applied = TRUE
                        WHERE id = ANY($1)
                    """, reward_ids)
                    
                    logger.info(f"Applied ${total_credits} in referral credits to org {organization_id}")
                
                return total_credits
    
    async def get_referral_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get top referrers leaderboard"""
        async with self.db_pool.acquire() as conn:
            leaderboard = await conn.fetch("""
                SELECT 
                    rs.organization_id,
                    rs.successful_referrals,
                    rs.total_rewards_earned,
                    o.name as organization_name
                FROM referral_stats rs
                LEFT JOIN organizations o ON rs.organization_id = o.id
                WHERE rs.successful_referrals > 0
                ORDER BY rs.successful_referrals DESC, rs.total_rewards_earned DESC
                LIMIT $1
            """, limit)
        
        return [dict(row) for row in leaderboard]

# Referral program configurations
REFERRAL_PROGRAMS = {
    "standard": {
        "reward_type": "fixed_credit",
        "reward_value": Decimal("25.00"),
        "referrer_reward": Decimal("25.00"),
        "referred_discount": Decimal("10.00"),  # 10% first month discount
        "max_uses": 50,
        "expires_days": 90
    },
    "enterprise": {
        "reward_type": "percentage_discount",
        "reward_value": Decimal("50.00"),
        "referrer_reward": Decimal("100.00"),
        "referred_discount": Decimal("20.00"),  # 20% first month discount
        "max_uses": 10,
        "expires_days": 180
    }
}