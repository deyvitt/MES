"""
Load Balancer for Mamba Swarm API
Distributes requests across multiple API server instances
"""

import asyncio
import aiohttp
import random
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import hashlib

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    HASH_BASED = "hash_based"
    RESOURCE_AWARE = "resource_aware"

@dataclass
class ServerInstance:
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    timeout: float = 30.0
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_health_check: float = 0.0
    is_healthy: bool = True
    health_check_failures: int = 0
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def avg_response_time(self) -> float:
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.total_requests
        if total == 0:
            return 1.0
        return (total - self.failed_requests) / total
    
    @property
    def load_score(self) -> float:
        """Calculate load score for resource-aware balancing"""
        connection_load = self.current_connections / self.max_connections
        response_time_load = min(self.avg_response_time / 1000.0, 1.0)  # Normalize to seconds
        failure_rate = self.failed_requests / max(self.total_requests, 1)
        
        return (connection_load * 0.4 + response_time_load * 0.4 + failure_rate * 0.2)

class LoadBalancer:
    """Advanced load balancer for Mamba Swarm API servers"""
    
    def __init__(self, 
                 servers: List[Tuple[str, int]], 
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE,
                 health_check_interval: float = 30.0,
                 health_check_timeout: float = 5.0,
                 max_retries: int = 3):
        
        self.logger = logging.getLogger(__name__)
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.max_retries = max_retries
        
        # Initialize server instances
        self.servers = [
            ServerInstance(host=host, port=port) 
            for host, port in servers
        ]
        
        # Strategy-specific state
        self.round_robin_index = 0
        self.request_counts = defaultdict(int)
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Health check task
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.total_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
    
    async def start(self):
        """Start the load balancer"""
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30.0, connect=10.0)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Start health check task
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Initial health check
        await self._check_all_servers_health()
        
        self.logger.info(f"Load balancer started with {len(self.servers)} servers using {self.strategy.value} strategy")
    
    async def stop(self):
        """Stop the load balancer"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        self.logger.info("Load balancer stopped")
    
    def get_healthy_servers(self) -> List[ServerInstance]:
        """Get list of healthy servers"""
        return [server for server in self.servers if server.is_healthy]
    
    def select_server(self, request_data: Optional[Dict[str, Any]] = None) -> Optional[ServerInstance]:
        """Select server based on configured strategy"""
        healthy_servers = self.get_healthy_servers()
        
        if not healthy_servers:
            self.logger.warning("No healthy servers available")
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.HASH_BASED:
            return self._hash_based_selection(healthy_servers, request_data)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._resource_aware_selection(healthy_servers)
        else:
            return random.choice(healthy_servers)
    
    def _round_robin_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Round-robin server selection"""
        server = servers[self.round_robin_index % len(servers)]
        self.round_robin_index += 1
        return server
    
    def _least_connections_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Select server with least connections"""
        return min(servers, key=lambda s: s.current_connections)
    
    def _weighted_round_robin_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Weighted round-robin selection"""
        total_weight = sum(s.weight for s in servers)
        random_weight = random.uniform(0, total_weight)
        
        current_weight = 0
        for server in servers:
            current_weight += server.weight
            if random_weight <= current_weight:
                return server
        
        return servers[-1]  # Fallback
    
    def _least_response_time_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Select server with least average response time"""
        return min(servers, key=lambda s: s.avg_response_time or float('inf'))
    
    def _hash_based_selection(self, servers: List[ServerInstance], request_data: Optional[Dict[str, Any]]) -> ServerInstance:
        """Hash-based selection for session affinity"""
        if not request_data or 'prompt' not in request_data:
            return random.choice(servers)
        
        # Use prompt hash for consistent routing
        prompt_hash = hashlib.md5(request_data['prompt'].encode()).hexdigest()
        server_index = int(prompt_hash, 16) % len(servers)
        return servers[server_index]
    
    def _resource_aware_selection(self, servers: List[ServerInstance]) -> ServerInstance:
        """Select server based on resource utilization"""
        # Sort by load score (lower is better)
        sorted_servers = sorted(servers, key=lambda s: s.load_score)
        
        # Use weighted random selection favoring lower load servers
        weights = [1.0 / (s.load_score + 0.1) for s in sorted_servers]
        total_weight = sum(weights)
        
        random_value = random.uniform(0, total_weight)
        current_weight = 0
        
        for server, weight in zip(sorted_servers, weights):
            current_weight += weight
            if random_value <= current_weight:
                return server
        
        return sorted_servers[0]  # Fallback to best server
    
    async def forward_request(self, 
                             path: str, 
                             method: str = "POST", 
                             data: Optional[Dict[str, Any]] = None,
                             headers: Optional[Dict[str, str]] = None,
                             **kwargs) -> Tuple[int, Dict[str, Any]]:
        """Forward request to selected server with retry logic"""
        self.total_requests += 1
        
        for attempt in range(self.max_retries + 1):
            server = self.select_server(data)
            if not server:
                self.failed_requests += 1
                return 503, {"error": "No healthy servers available"}
            
            try:
                start_time = time.time()
                server.current_connections += 1
                
                url = f"{server.url}{path}"
                request_kwargs = {
                    "timeout": aiohttp.ClientTimeout(total=server.timeout),
                    **kwargs
                }
                
                if headers:
                    request_kwargs["headers"] = headers
                
                if data:
                    request_kwargs["json"] = data
                
                async with self.session.request(method, url, **request_kwargs) as response:
                    response_time = time.time() - start_time
                    response_data = await response.json()
                    
                    # Update server metrics
                    server.current_connections -= 1
                    server.total_requests += 1
                    server.response_times.append(response_time * 1000)  # Store in ms
                    
                    if response.status >= 400:
                        server.failed_requests += 1
                        
                        if attempt < self.max_retries:
                            self.logger.warning(f"Request failed on {server.url} (attempt {attempt + 1}), retrying...")
                            continue
                    
                    return response.status, response_data
                    
            except Exception as e:
                server.current_connections = max(0, server.current_connections - 1)
                server.failed_requests += 1
                
                self.logger.error(f"Request failed on {server.url}: {e}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
        
        self.failed_requests += 1
        return 502, {"error": "All servers failed after retries"}
    
    async def _check_server_health(self, server: ServerInstance) -> bool:
        """Check health of a single server"""
        try:
            url = f"{server.url}/health"
            timeout = aiohttp.ClientTimeout(total=self.health_check_timeout)
            
            async with self.session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    health_data = await response.json()
                    server.last_health_check = time.time()
                    server.health_check_failures = 0
                    
                    # Update server metrics from health data if available
                    if 'system_info' in health_data:
                        # Could extract additional metrics here
                        pass
                    
                    return True
                else:
                    server.health_check_failures += 1
                    return False
                    
        except Exception as e:
            server.health_check_failures += 1
            self.logger.debug(f"Health check failed for {server.url}: {e}")
            return False
    
    async def _check_all_servers_health(self):
        """Check health of all servers"""
        tasks = [self._check_server_health(server) for server in self.servers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for server, result in zip(self.servers, results):
            if isinstance(result, Exception):
                server.is_healthy = False
                server.health_check_failures += 1
            else:
                was_healthy = server.is_healthy
                server.is_healthy = result and server.health_check_failures < 3
                
                if not was_healthy and server.is_healthy:
                    self.logger.info(f"Server {server.url} is back online")
                elif was_healthy and not server.is_healthy:
                    self.logger.warning(f"Server {server.url} is unhealthy")
    
    async def _health_check_loop(self):
        """Periodic health check loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_servers_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
    
    def add_server(self, host: str, port: int, weight: float = 1.0):
        """Add a new server to the pool"""
        server = ServerInstance(host=host, port=port, weight=weight)
        self.servers.append(server)
        self.logger.info(f"Added server {server.url}")
    
    def remove_server(self, host: str, port: int):
        """Remove a server from the pool"""
        self.servers = [s for s in self.servers if not (s.host == host and s.port == port)]
        self.logger.info(f"Removed server http://{host}:{port}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        uptime = time.time() - self.start_time
        
        server_stats = []
        for server in self.servers:
            server_stats.append({
                "url": server.url,
                "is_healthy": server.is_healthy,
                "current_connections": server.current_connections,
                "total_requests": server.total_requests,
                "failed_requests": server.failed_requests,
                "success_rate": server.success_rate,
                "avg_response_time_ms": server.avg_response_time,
                "load_score": server.load_score,
                "weight": server.weight
            })
        
        return {
            "strategy": self.strategy.value,
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.total_requests - self.failed_requests) / max(self.total_requests, 1),
            "healthy_servers": len(self.get_healthy_servers()),
            "total_servers": len(self.servers),
            "servers": server_stats
        }

# FastAPI integration
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

def create_load_balancer_app(servers: List[Tuple[str, int]], 
                           strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE) -> FastAPI:
    """Create FastAPI app with load balancer"""
    
    app = FastAPI(title="Mamba Swarm Load Balancer", version="1.0.0")
    load_balancer = LoadBalancer(servers, strategy)
    
    @app.on_event("startup")
    async def startup():
        await load_balancer.start()
    
    @app.on_event("shutdown")
    async def shutdown():
        await load_balancer.stop()
    
    @app.get("/lb/health")
    async def lb_health():
        """Load balancer health endpoint"""
        return {"status": "healthy", "stats": load_balancer.get_stats()}
    
    @app.get("/lb/stats")
    async def lb_stats():
        """Get load balancer statistics"""
        return load_balancer.get_stats()
    
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_request(request: Request, path: str):
        """Proxy all requests to backend servers"""
        try:
            # Get request data
            body = await request.body()
            headers = dict(request.headers)
            
            # Remove hop-by-hop headers
            headers.pop("host", None)
            headers.pop("connection", None)
            
            # Parse body if it's JSON
            data = None
            if body:
                try:
                    import json
                    data = json.loads(body.decode())
                except:
                    pass
            
            # Forward request
            status, response_data = await load_balancer.forward_request(
                f"/{path}",
                request.method,
                data=data,
                headers=headers,
                params=dict(request.query_params)
            )
            
            return JSONResponse(content=response_data, status_code=status)
            
        except Exception as e:
            return JSONResponse(
                content={"error": f"Load balancer error: {str(e)}"},
                status_code=500
            )
    
    return app

def run_load_balancer(servers: List[Tuple[str, int]], 
                     host: str = "0.0.0.0", 
                     port: int = 8080,
                     strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE):
    """Run the load balancer"""
    app = create_load_balancer_app(servers, strategy)
    
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    # Example usage
    servers = [
        ("localhost", 8000),
        ("localhost", 8001),
        ("localhost", 8002),
    ]
    
    run_load_balancer(servers, strategy=LoadBalancingStrategy.RESOURCE_AWARE) 