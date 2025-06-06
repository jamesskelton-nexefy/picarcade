import pytest
import httpx
from datetime import datetime
from typing import Dict, Any

class TestHealthEndpoint:
    """
    Phase 1: Health check tests with real API calls
    Tests that /api/health returns version metadata
    """
    
    @pytest.mark.asyncio
    async def test_health_check_real_api(self):
        """
        Test health check hits /api/health and returns version metadata
        This is a real API test as specified in Phase 1
        """
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/api/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            assert "status" in data
            assert "timestamp" in data
            assert "version" in data
            assert "environment" in data
            assert "services" in data
            
            # Validate specific values
            assert data["status"] == "healthy"
            assert data["version"] == "0.1.0"
            assert data["environment"] in ["development", "staging", "production"]
            
            # Validate timestamp is recent (within last minute)
            timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
            now = datetime.now(timestamp.tzinfo)
            time_diff = abs((now - timestamp).total_seconds())
            assert time_diff < 60  # Within last minute
            
            # Validate services structure
            services = data["services"]
            assert isinstance(services, dict)
            assert "api" in services
            assert services["api"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_version_endpoint(self):
        """Test version endpoint returns correct metadata"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/api/version")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "version" in data
            assert "build_time" in data
            assert "environment" in data
            assert data["version"] == "0.1.0"
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Test root endpoint is accessible"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/")
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data 