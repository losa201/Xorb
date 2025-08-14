"""
Secure Query Builder
Production-grade SQL query builder with injection prevention and tenant isolation
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from uuid import UUID
from enum import Enum
from dataclasses import dataclass

from sqlalchemy import text, select, update, delete, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import ClauseElement

from ..core.secure_tenant_context import TenantContext
from ..core.logging import get_logger

logger = get_logger(__name__)


class QueryType(str, Enum):
    """Supported query types"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class SecurityLevel(str, Enum):
    """Query security levels"""
    STRICT = "strict"      # Maximum security, all validation enabled
    STANDARD = "standard"  # Standard validation
    RELAXED = "relaxed"    # Minimal validation (for admin operations)


@dataclass
class QueryValidationResult:
    """Result of query security validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized_query: Optional[str] = None


@dataclass
class SecureQueryParams:
    """Container for secure query parameters"""
    query: str
    params: Dict[str, Any]
    tenant_id: UUID
    query_type: QueryType
    security_level: SecurityLevel = SecurityLevel.STANDARD


class SecureQueryBuilder:
    """
    Production-grade secure query builder
    
    Features:
    - SQL injection prevention through parameterization
    - Mandatory tenant isolation for all queries
    - Query validation and sanitization
    - Performance monitoring and caching
    - Audit logging for all queries
    """
    
    def __init__(self, tenant_context: TenantContext):
        self.tenant_context = tenant_context
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Security patterns - queries matching these are blocked
        self.dangerous_patterns = [
            r'\b(DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE)\b',  # DDL
            r'\b(EXEC|EXECUTE|xp_|sp_)\b',                     # Stored procedures
            r'(\|\||&&|\|\|)',                                 # Command injection
            r'(\-\-|\/\*|\*\/)',                              # SQL comments
            r'\b(UNION|UNION\s+ALL)\b(?![^\']*\')',           # Union attacks
            r';\s*\w+',                                        # Multiple statements
            r'\b(INFORMATION_SCHEMA|sys\.)\b',                 # System tables
            r'(\bOR\b.*=.*\bOR\b)',                           # Classic injection
            r'(\'\s*(OR|AND)\s*\'\s*=\s*\')',                # Quote manipulation
        ]
        
        # Compile patterns for performance
        self.dangerous_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
        
        # Tables that require tenant isolation
        self.tenant_tables = {
            'tenants', 'tenant_users', 'findings', 'evidence', 
            'scan_sessions', 'scan_results', 'embedding_vectors',
            'threat_indicators', 'security_incidents', 'audit_logs'
        }
        
        # Allowed system tables for read-only operations
        self.allowed_system_tables = {
            'pg_tables', 'pg_class', 'pg_attribute', 'pg_index'
        }
    
    def build_select(
        self, 
        table: str,
        columns: List[str] = None,
        where_conditions: Dict[str, Any] = None,
        joins: List[Dict[str, str]] = None,
        order_by: List[str] = None,
        limit: int = None,
        offset: int = None
    ) -> SecureQueryParams:
        """
        Build secure SELECT query with tenant isolation
        
        Args:
            table: Primary table name
            columns: Columns to select (default: all)
            where_conditions: WHERE clause conditions
            joins: List of join specifications
            order_by: ORDER BY columns
            limit: LIMIT value
            offset: OFFSET value
            
        Returns:
            SecureQueryParams: Validated query with parameters
        """
        # Validate table name
        self._validate_table_name(table)
        
        # Build column list
        if columns:
            column_list = ", ".join([self._sanitize_identifier(col) for col in columns])
        else:
            column_list = "*"
        
        # Start building query
        query_parts = [f"SELECT {column_list}"]
        query_parts.append(f"FROM {self._sanitize_identifier(table)}")
        
        # Build parameters
        params = {"tenant_id": str(self.tenant_context.tenant_id)}
        
        # Add joins
        if joins:
            for join in joins:
                join_type = join.get('type', 'INNER')
                join_table = self._sanitize_identifier(join['table'])
                join_condition = join['condition']  # Should be parameterized
                query_parts.append(f"{join_type} JOIN {join_table} ON {join_condition}")
        
        # Build WHERE clause with tenant isolation
        where_clauses = []
        
        # Add tenant isolation for tenant-scoped tables
        if table in self.tenant_tables:
            where_clauses.append(f"{self._sanitize_identifier(table)}.tenant_id = :tenant_id")
        
        # Add custom conditions
        if where_conditions:
            for field, value in where_conditions.items():
                param_name = f"param_{len(params)}"
                where_clauses.append(f"{self._sanitize_identifier(field)} = :{param_name}")
                params[param_name] = value
        
        if where_clauses:
            query_parts.append(f"WHERE {' AND '.join(where_clauses)}")
        
        # Add ORDER BY
        if order_by:
            sanitized_order = [self._sanitize_identifier(col) for col in order_by]
            query_parts.append(f"ORDER BY {', '.join(sanitized_order)}")
        
        # Add LIMIT and OFFSET
        if limit:
            query_parts.append(f"LIMIT {int(limit)}")
        if offset:
            query_parts.append(f"OFFSET {int(offset)}")
        
        query = " ".join(query_parts)
        
        return SecureQueryParams(
            query=query,
            params=params,
            tenant_id=self.tenant_context.tenant_id,
            query_type=QueryType.SELECT
        )
    
    def build_insert(
        self, 
        table: str,
        data: Dict[str, Any],
        returning: List[str] = None
    ) -> SecureQueryParams:
        """
        Build secure INSERT query with tenant isolation
        
        Args:
            table: Table name
            data: Data to insert
            returning: Columns to return
            
        Returns:
            SecureQueryParams: Validated query with parameters
        """
        self._validate_table_name(table)
        
        # Ensure tenant_id is included for tenant-scoped tables
        if table in self.tenant_tables:
            data['tenant_id'] = str(self.tenant_context.tenant_id)
        
        # Build column and value lists
        columns = list(data.keys())
        sanitized_columns = [self._sanitize_identifier(col) for col in columns]
        placeholders = [f":{col}" for col in columns]
        
        query_parts = [
            f"INSERT INTO {self._sanitize_identifier(table)}",
            f"({', '.join(sanitized_columns)})",
            f"VALUES ({', '.join(placeholders)})"
        ]
        
        # Add RETURNING clause
        if returning:
            sanitized_returning = [self._sanitize_identifier(col) for col in returning]
            query_parts.append(f"RETURNING {', '.join(sanitized_returning)}")
        
        query = " ".join(query_parts)
        
        return SecureQueryParams(
            query=query,
            params=data,
            tenant_id=self.tenant_context.tenant_id,
            query_type=QueryType.INSERT
        )
    
    def build_update(
        self, 
        table: str,
        data: Dict[str, Any],
        where_conditions: Dict[str, Any],
        returning: List[str] = None
    ) -> SecureQueryParams:
        """
        Build secure UPDATE query with tenant isolation
        
        Args:
            table: Table name
            data: Data to update
            where_conditions: WHERE clause conditions
            returning: Columns to return
            
        Returns:
            SecureQueryParams: Validated query with parameters
        """
        self._validate_table_name(table)
        
        # Prevent tenant_id modification
        if 'tenant_id' in data:
            self.logger.warning(f"Attempted tenant_id modification blocked in UPDATE on {table}")
            del data['tenant_id']
        
        if not data:
            raise ValueError("No valid data to update")
        
        # Build SET clause
        set_clauses = []
        params = {}
        
        for field, value in data.items():
            param_name = f"set_{field}"
            set_clauses.append(f"{self._sanitize_identifier(field)} = :{param_name}")
            params[param_name] = value
        
        query_parts = [
            f"UPDATE {self._sanitize_identifier(table)}",
            f"SET {', '.join(set_clauses)}"
        ]
        
        # Build WHERE clause with tenant isolation
        where_clauses = []
        
        # Add tenant isolation for tenant-scoped tables
        if table in self.tenant_tables:
            where_clauses.append(f"tenant_id = :tenant_id")
            params['tenant_id'] = str(self.tenant_context.tenant_id)
        
        # Add custom conditions
        for field, value in where_conditions.items():
            param_name = f"where_{field}"
            where_clauses.append(f"{self._sanitize_identifier(field)} = :{param_name}")
            params[param_name] = value
        
        if where_clauses:
            query_parts.append(f"WHERE {' AND '.join(where_clauses)}")
        else:
            raise ValueError("UPDATE queries must include WHERE conditions")
        
        # Add RETURNING clause
        if returning:
            sanitized_returning = [self._sanitize_identifier(col) for col in returning]
            query_parts.append(f"RETURNING {', '.join(sanitized_returning)}")
        
        query = " ".join(query_parts)
        
        return SecureQueryParams(
            query=query,
            params=params,
            tenant_id=self.tenant_context.tenant_id,
            query_type=QueryType.UPDATE
        )
    
    def build_delete(
        self, 
        table: str,
        where_conditions: Dict[str, Any]
    ) -> SecureQueryParams:
        """
        Build secure DELETE query with tenant isolation
        
        Args:
            table: Table name
            where_conditions: WHERE clause conditions
            
        Returns:
            SecureQueryParams: Validated query with parameters
        """
        self._validate_table_name(table)
        
        params = {}
        where_clauses = []
        
        # Add tenant isolation for tenant-scoped tables
        if table in self.tenant_tables:
            where_clauses.append(f"tenant_id = :tenant_id")
            params['tenant_id'] = str(self.tenant_context.tenant_id)
        
        # Add custom conditions
        for field, value in where_conditions.items():
            param_name = f"where_{field}"
            where_clauses.append(f"{self._sanitize_identifier(field)} = :{param_name}")
            params[param_name] = value
        
        if not where_clauses:
            raise ValueError("DELETE queries must include WHERE conditions")
        
        query = f"DELETE FROM {self._sanitize_identifier(table)} WHERE {' AND '.join(where_clauses)}"
        
        return SecureQueryParams(
            query=query,
            params=params,
            tenant_id=self.tenant_context.tenant_id,
            query_type=QueryType.DELETE
        )
    
    def validate_query(
        self, 
        query: str, 
        security_level: SecurityLevel = SecurityLevel.STANDARD
    ) -> QueryValidationResult:
        """
        Validate query for security issues
        
        Args:
            query: SQL query to validate
            security_level: Security level for validation
            
        Returns:
            QueryValidationResult: Validation results
        """
        errors = []
        warnings = []
        
        # Check for dangerous patterns
        for pattern in self.dangerous_regex:
            if pattern.search(query):
                errors.append(f"Query contains dangerous pattern: {pattern.pattern}")
        
        # Check for tenant isolation on tenant-scoped tables
        if security_level in [SecurityLevel.STRICT, SecurityLevel.STANDARD]:
            normalized_query = query.lower()
            
            for table in self.tenant_tables:
                if table in normalized_query:
                    if 'tenant_id' not in normalized_query:
                        errors.append(f"Query accessing {table} must include tenant isolation")
        
        # Check for parameterization
        if "'" in query and ':' not in query:
            warnings.append("Query may contain string literals instead of parameters")
        
        # Check for multiple statements
        if ';' in query.strip().rstrip(';'):
            errors.append("Multiple statements not allowed")
        
        # Normalize query for caching
        normalized_query = re.sub(r'\s+', ' ', query.strip())
        
        return QueryValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_query=normalized_query
        )
    
    def _validate_table_name(self, table: str) -> None:
        """Validate table name to prevent injection"""
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table):
            raise ValueError(f"Invalid table name: {table}")
        
        # Check if table is allowed
        if table not in self.tenant_tables and table not in self.allowed_system_tables:
            self.logger.warning(f"Access to potentially restricted table: {table}")
    
    def _sanitize_identifier(self, identifier: str) -> str:
        """Sanitize SQL identifier (table, column names)"""
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            raise ValueError(f"Invalid identifier: {identifier}")
        return identifier
    
    async def execute_secure_query(
        self, 
        session: AsyncSession,
        query_params: SecureQueryParams,
        security_level: SecurityLevel = SecurityLevel.STANDARD
    ) -> Any:
        """
        Execute query with security validation
        
        Args:
            session: Database session
            query_params: Secure query parameters
            security_level: Security level for validation
            
        Returns:
            Query result
        """
        # Validate query
        validation = self.validate_query(query_params.query, security_level)
        if not validation.is_valid:
            self.logger.error(f"Query validation failed: {validation.errors}")
            raise ValueError(f"Query validation failed: {'; '.join(validation.errors)}")
        
        # Log warnings
        for warning in validation.warnings:
            self.logger.warning(f"Query warning: {warning}")
        
        try:
            # Execute query
            result = await session.execute(text(query_params.query), query_params.params)
            
            # Log successful execution
            self.logger.info(
                f"Executed {query_params.query_type.value} query: "
                f"tenant={query_params.tenant_id}, "
                f"params={len(query_params.params)}"
            )
            
            return result
            
        except SQLAlchemyError as e:
            self.logger.error(f"Query execution failed: {e}")
            raise


# Utility functions for common secure operations

async def secure_select(
    session: AsyncSession,
    tenant_context: TenantContext,
    table: str,
    columns: List[str] = None,
    where_conditions: Dict[str, Any] = None,
    **kwargs
) -> Any:
    """Execute secure SELECT query"""
    builder = SecureQueryBuilder(tenant_context)
    query_params = builder.build_select(table, columns, where_conditions, **kwargs)
    return await builder.execute_secure_query(session, query_params)


async def secure_insert(
    session: AsyncSession,
    tenant_context: TenantContext,
    table: str,
    data: Dict[str, Any],
    returning: List[str] = None
) -> Any:
    """Execute secure INSERT query"""
    builder = SecureQueryBuilder(tenant_context)
    query_params = builder.build_insert(table, data, returning)
    return await builder.execute_secure_query(session, query_params)


async def secure_update(
    session: AsyncSession,
    tenant_context: TenantContext,
    table: str,
    data: Dict[str, Any],
    where_conditions: Dict[str, Any],
    returning: List[str] = None
) -> Any:
    """Execute secure UPDATE query"""
    builder = SecureQueryBuilder(tenant_context)
    query_params = builder.build_update(table, data, where_conditions, returning)
    return await builder.execute_secure_query(session, query_params)


async def secure_delete(
    session: AsyncSession,
    tenant_context: TenantContext,
    table: str,
    where_conditions: Dict[str, Any]
) -> Any:
    """Execute secure DELETE query"""
    builder = SecureQueryBuilder(tenant_context)
    query_params = builder.build_delete(table, where_conditions)
    return await builder.execute_secure_query(session, query_params)