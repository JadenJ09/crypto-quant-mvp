#!/bin/bash

# TimescaleDB Integration Test Script for Custom Backtest Engine
# This script starts the services and runs comprehensive integration tests

set -e

echo "🚀 Starting TimescaleDB Integration Test for Custom Backtest Engine"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "docker-compose.dev.yml" ]; then
    echo -e "${RED}❌ Error: docker-compose.dev.yml not found. Please run from project root.${NC}"
    exit 1
fi

# Function to check if service is healthy
check_service_health() {
    local service_name=$1
    local health_url=$2
    local max_attempts=30
    local attempt=0

    echo -e "${BLUE}🔍 Checking $service_name health...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f -s "$health_url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name is healthy${NC}"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo -e "${YELLOW}⏳ Waiting for $service_name (attempt $attempt/$max_attempts)...${NC}"
        sleep 5
    done
    
    echo -e "${RED}❌ $service_name failed to become healthy${NC}"
    return 1
}

# Start core services
echo -e "${BLUE}🔄 Starting core services...${NC}"
docker compose -f docker-compose.dev.yml up -d timescaledb kafka

echo -e "${BLUE}⏳ Waiting for core services to be ready...${NC}"
sleep 15

# Check TimescaleDB health
if ! docker compose -f docker-compose.dev.yml exec -T timescaledb pg_isready -h localhost -p 5433 -U quant_user; then
    echo -e "${RED}❌ TimescaleDB is not ready${NC}"
    docker compose -f docker-compose.dev.yml logs timescaledb
    exit 1
fi

echo -e "${GREEN}✅ TimescaleDB is ready${NC}"

# Start backtest engine
echo -e "${BLUE}🔄 Starting backtest engine...${NC}"
docker compose -f docker-compose.dev.yml up -d --build backtest-engine

# Wait for backtest engine to be ready
if ! check_service_health "Backtest Engine" "http://localhost:8003/health/"; then
    echo -e "${RED}❌ Backtest engine failed to start${NC}"
    docker compose -f docker-compose.dev.yml logs backtest-engine
    exit 1
fi

# Show service status
echo -e "${BLUE}📊 Service Status:${NC}"
docker compose -f docker-compose.dev.yml ps

# Run integration tests
echo -e "${BLUE}🧪 Running integration tests...${NC}"

# Navigate to backtest engine directory
cd services/backtest-engine

# Check if Python dependencies are available
if ! python3 -c "import requests, asyncio" 2>/dev/null; then
    echo -e "${YELLOW}⚠️ Installing Python dependencies...${NC}"
    pip3 install requests asyncio
fi

# Run the integration test
echo -e "${BLUE}🔄 Executing comprehensive integration test...${NC}"
python3 run_integration_tests.py --output integration_test_results.json

# Check test results
if [ -f "integration_test_results.json" ]; then
    echo -e "${GREEN}✅ Test results saved to integration_test_results.json${NC}"
    
    # Show summary
    echo -e "${BLUE}📋 Test Summary:${NC}"
    python3 -c "
import json
with open('integration_test_results.json', 'r') as f:
    results = json.load(f)
    summary = results['summary']
    print(f'Total Tests: {summary[\"total_tests\"]}')
    print(f'Passed: {summary[\"passed\"]}')
    print(f'Failed: {summary[\"failed\"]}')
    print(f'Success Rate: {summary[\"success_rate\"]:.1f}%')
    
    # Show failed tests
    if summary['failed'] > 0:
        print('\n❌ Failed Tests:')
        for test_name, result in results['results'].items():
            if not result['success']:
                print(f'  - {test_name}: {result.get(\"error\", \"Unknown error\")}')
"
else
    echo -e "${RED}❌ Test results file not found${NC}"
fi

# Go back to project root
cd ../..

echo -e "${BLUE}🎯 Integration test completed!${NC}"
echo -e "${BLUE}📝 Services are running. You can now:${NC}"
echo -e "   • View API docs: ${BLUE}http://localhost:8003/docs${NC}"
echo -e "   • Check health: ${BLUE}http://localhost:8003/health/${NC}"
echo -e "   • TimescaleDB API: ${BLUE}http://localhost:8003/timescale/health${NC}"
echo -e "   • Stop services: ${YELLOW}docker compose -f docker-compose.dev.yml down${NC}"

echo -e "${GREEN}🎉 Custom Backtest Engine is ready for production!${NC}"
