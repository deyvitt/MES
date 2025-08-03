"""
API Server for Mamba Swarm
FastAPI-based server for serving the distributed Mamba language model
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
import asyncio
import json
import time
import logging
import torch
from contextlib import asynccontextmanager
import uvicorn

# Import your swarm components
from system.mambaSwarm import SwarmEngine
from system.inference import InferenceEngine
from routing.router import Router
from training.trainer import setup_logging

# Pydantic models for API
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_length: int = Field(default=100, ge=1, le=2048, description="Maximum generation length")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    stream: bool = Field(default=False, description="Enable streaming response")
    domain: Optional[str] = Field(default=None, description="Specific domain for routing")

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    generation_time: float
    tokens_generated: int
    model_info: Dict[str, Any]

class StreamingToken(BaseModel):
    token: str
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    swarm_status: Dict[str, Any]
    system_info: Dict[str, Any]
    timestamp: float

class ModelInfo(BaseModel):
    total_parameters: int
    active_encoders: int
    total_encoders: int
    memory_usage: Dict[str, float]
    device_info: List[str]

# Global swarm engine instance
swarm_engine: Optional[SwarmEngine] = None
inference_engine: Optional[InferenceEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global swarm_engine, inference_engine
    
    # Startup
    logging.info("Initializing Mamba Swarm API Server...")
    
    try:
        # Initialize swarm engine
        swarm_engine = SwarmEngine()
        await asyncio.get_event_loop().run_in_executor(None, swarm_engine.initialize)
        
        # Initialize inference engine
        inference_engine = InferenceEngine(swarm_engine)
        
        logging.info("Mamba Swarm API Server initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize swarm: {e}")
        raise
    
    yield
    
    # Shutdown
    logging.info("Shutting down Mamba Swarm API Server...")
    if swarm_engine:
        swarm_engine.shutdown()

# Create FastAPI app
app = FastAPI(
    title="Mamba Swarm API",
    description="Distributed Mamba Language Model API with 100 encoder units",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get swarm engine
async def get_swarm_engine() -> SwarmEngine:
    if swarm_engine is None:
        raise HTTPException(status_code=503, detail="Swarm engine not initialized")
    return swarm_engine

async def get_inference_engine() -> InferenceEngine:
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    return inference_engine

@app.get("/health", response_model=HealthResponse)
async def health_check(swarm: SwarmEngine = Depends(get_swarm_engine)):
    """Health check endpoint"""
    try:
        swarm_status = swarm.get_status()
        system_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "python_version": "3.8+",
        }
        
        return HealthResponse(
            status="healthy",
            swarm_status=swarm_status,
            system_info=system_info,
            timestamp=time.time()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(swarm: SwarmEngine = Depends(get_swarm_engine)):
    """Get model information"""
    try:
        info = swarm.get_model_info()
        memory_stats = swarm.memory_manager.get_memory_stats()
        
        return ModelInfo(
            total_parameters=info.get("total_parameters", 7000000000),  # 100 * 70M
            active_encoders=info.get("active_encoders", 100),
            total_encoders=info.get("total_encoders", 100),
            memory_usage={
                "system_memory_gb": memory_stats.used_memory,
                "gpu_memory_gb": memory_stats.gpu_memory,
                "cache_size_gb": memory_stats.cache_size
            },
            device_info=info.get("devices", ["cuda:0" if torch.cuda.is_available() else "cpu"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    inference: InferenceEngine = Depends(get_inference_engine)
):
    """Generate text from prompt"""
    try:
        start_time = time.time()
        
        # Generate text
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            inference.generate,
            request.prompt,
            {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "domain": request.domain
            }
        )
        
        generation_time = time.time() - start_time
        
        return GenerationResponse(
            generated_text=result["generated_text"],
            prompt=request.prompt,
            generation_time=generation_time,
            tokens_generated=result.get("tokens_generated", 0),
            model_info=result.get("model_info", {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/stream")
async def generate_text_stream(
    request: GenerationRequest,
    inference: InferenceEngine = Depends(get_inference_engine)
):
    """Generate text with streaming response"""
    if not request.stream:
        raise HTTPException(status_code=400, detail="Streaming not requested")
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Create generator for streaming
            generator = inference.generate_stream(
                request.prompt,
                {
                    "max_length": request.max_length,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "repetition_penalty": request.repetition_penalty,
                    "domain": request.domain
                }
            )
            
            for token_data in generator:
                streaming_token = StreamingToken(
                    token=token_data.get("token", ""),
                    is_final=token_data.get("is_final", False),
                    metadata=token_data.get("metadata", {})
                )
                
                yield f"data: {streaming_token.json()}\n\n"
                
                if streaming_token.is_final:
                    break
            
        except Exception as e:
            error_token = StreamingToken(
                token="",
                is_final=True,
                metadata={"error": str(e)}
            )
            yield f"data: {error_token.json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.post("/generate/batch")
async def generate_batch(
    requests: List[GenerationRequest],
    inference: InferenceEngine = Depends(get_inference_engine)
):
    """Generate text for multiple prompts"""
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Batch size too large (max 10)")
    
    try:
        # Process requests in parallel
        tasks = []
        for req in requests:
            task = asyncio.get_event_loop().run_in_executor(
                None,
                inference.generate,
                req.prompt,
                {
                    "max_length": req.max_length,
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "top_k": req.top_k,
                    "repetition_penalty": req.repetition_penalty,
                    "domain": req.domain
                }
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = []
        for i, (req, result) in enumerate(zip(requests, results)):
            if isinstance(result, Exception):
                responses.append({
                    "error": str(result),
                    "prompt": req.prompt,
                    "index": i
                })
            else:
                responses.append(GenerationResponse(
                    generated_text=result["generated_text"],
                    prompt=req.prompt,
                    generation_time=result.get("generation_time", 0),
                    tokens_generated=result.get("tokens_generated", 0),
                    model_info=result.get("model_info", {})
                ))
        
        return {"responses": responses}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

@app.get("/metrics")
async def get_metrics(swarm: SwarmEngine = Depends(get_swarm_engine)):
    """Get system metrics"""
    try:
        metrics = {
            "memory_report": swarm.memory_manager.get_memory_report(),
            "swarm_metrics": swarm.get_metrics(),
            "inference_stats": swarm.get_inference_stats() if hasattr(swarm, 'get_inference_stats') else {},
            "timestamp": time.time()
        }
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.post("/admin/reload")
async def reload_model(
    background_tasks: BackgroundTasks,
    swarm: SwarmEngine = Depends(get_swarm_engine)
):
    """Reload the model (admin endpoint)"""
    try:
        background_tasks.add_task(swarm.reload_model)
        return {"message": "Model reload initiated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.post("/admin/cleanup")
async def cleanup_memory(swarm: SwarmEngine = Depends(get_swarm_engine)):
    """Force memory cleanup (admin endpoint)"""
    try:
        swarm.memory_manager.cleanup_memory(aggressive=True)
        return {"message": "Memory cleanup completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup memory: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": time.time()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": time.time()
    }

def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the API server"""
    setup_logging()
    
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        reload=False  # Set to True for development
    )
    
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    run_server() 