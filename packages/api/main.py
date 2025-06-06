from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from datetime import datetime
import uvicorn
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure comprehensive logging to see all workflow details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set specific loggers to INFO level to see detailed workflow logs
logging.getLogger('pic_arcade_agentic').setLevel(logging.INFO)
logging.getLogger('pic_arcade_agentic.tools').setLevel(logging.INFO)
logging.getLogger('pic_arcade_agentic.tools.workflow_tools').setLevel(logging.INFO)
logging.getLogger('pic_arcade_agentic.tools.image_tools').setLevel(logging.INFO)
logging.getLogger('pic_arcade_agentic.agents').setLevel(logging.INFO)
logging.getLogger('pic_arcade_agentic.agents.mem0_tool_agent').setLevel(logging.INFO)

print("🔧 LOGGING CONFIGURED - Detailed workflow logs enabled")

# Add the agentic package to the path
agentic_path = Path(__file__).parent.parent / "agentic" / "src"
sys.path.append(str(agentic_path))

from pic_arcade_agentic.agents.mem0_tool_agent import Mem0ToolFirstAgent
from pic_arcade_agentic.types import OpenAIConfig

# Initialize FastAPI app
app = FastAPI(
    title="Pic Arcade API",
    description="Agentic AI-powered image and video generation platform with Mem0 persistent memory",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://localhost:3008",  # Added for current frontend port
        "http://localhost:3009",  # Added for current frontend port
        "http://localhost:3010",  # Added for current frontend port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001", 
        "http://127.0.0.1:3008",  # Added for current frontend port
        "http://127.0.0.1:3009",  # Added for current frontend port
        "http://127.0.0.1:3010"   # Added for current frontend port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check response model
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str
    services: Dict[str, str]

# Request/Response models for tool-first workflow
class PromptRequest(BaseModel):
    prompt: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None  # Added for Mem0 persistent memory

class WorkflowStep(BaseModel):
    step: int
    tool_name: str
    description: str
    inputs: Dict[str, Any]
    expected_output: str
    dependencies: List[str]

class WorkflowPlan(BaseModel):
    workflow_plan: List[WorkflowStep]
    reasoning: str
    confidence: float
    estimated_time: Optional[float] = None
    estimated_cost: Optional[float] = None

class ExecutionResult(BaseModel):
    step: int
    tool_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    resolved_inputs: Optional[Dict[str, Any]] = None

class ExecutionData(BaseModel):
    execution_results: List[ExecutionResult]
    final_outputs: Dict[str, Any]
    execution_status: str
    total_time: float
    errors: List[str]

class WorkflowMetadata(BaseModel):
    tools_used: List[str]
    total_time: float
    execution_status: str

class ToolFirstResponse(BaseModel):
    success: bool
    request_id: str
    user_request: str
    workflow_plan: Optional[WorkflowPlan] = None
    execution_results: Optional[ExecutionData] = None
    metadata: Optional[WorkflowMetadata] = None
    error: Optional[str] = None

# Initialize tool-first agent
tool_first_agent = Mem0ToolFirstAgent()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Pic Arcade API"}

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for Phase 1 testing.
    Returns version metadata and service status.
    """
    try:
        services = {
            "api": "healthy",
            "database": "checking...",  # Will implement proper DB check later
            "redis": "checking...",     # Will implement proper Redis check later
            "agentic": "healthy"        # Will implement proper agentic service check later
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="0.1.0",
            environment=os.getenv("ENVIRONMENT", "development"),
            services=services
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/api/version")
async def get_version():
    """Get API version information"""
    return {
        "version": "0.1.0",
        "build_time": datetime.utcnow().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.post("/api/workflow/process", response_model=ToolFirstResponse)
async def process_prompt(request: PromptRequest):
    """
    Process a user prompt through the Mem0-enhanced Tool-First AI agent workflow.
    Returns dynamic workflow planning and execution results with persistent memory.
    """
    try:
        # Generate request ID if not provided
        request_id = request.request_id or f"req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate or use provided user_id for Mem0 memory persistence
        user_id = request.user_id or f"user_{datetime.utcnow().strftime('%Y%m%d')}"  # Default daily user ID
        
        print("=" * 100)
        print("🌐 API ENDPOINT: /api/workflow/process")
        print("=" * 100)
        print(f"📝 ORIGINAL USER PROMPT: '{request.prompt}'")
        print(f"🆔 REQUEST ID: {request_id}")
        print(f"👤 USER ID: {user_id}")
        print("=" * 100)
        
        # Process the prompt through the Mem0-enhanced tool-first agent
        result = await tool_first_agent.process_request(
            user_request=request.prompt,
            user_id=user_id  # Enable persistent memory across requests
        )
        
        # Create response based on agent result
        response = ToolFirstResponse(
            success=result["success"],
            request_id=request_id,
            user_request=request.prompt,
            error=result.get("error")
        )
        
        if result["success"]:
            # Add workflow plan
            if "workflow_plan" in result:
                plan_data = result["workflow_plan"]
                response.workflow_plan = WorkflowPlan(
                    workflow_plan=[
                        WorkflowStep(
                            step=step["step"],
                            tool_name=step["tool_name"],
                            description=step["description"],
                            inputs=step["inputs"],
                            expected_output=step["expected_output"],
                            dependencies=step.get("dependencies", [])
                        )
                        for step in plan_data.get("workflow_plan", [])
                    ],
                    reasoning=plan_data.get("reasoning", ""),
                    confidence=plan_data.get("confidence", 0.0),
                    estimated_time=plan_data.get("estimated_time"),
                    estimated_cost=plan_data.get("estimated_cost")
                )
            
            # Add execution results
            if "execution_results" in result:
                exec_data = result["execution_results"]
                response.execution_results = ExecutionData(
                    execution_results=[
                        ExecutionResult(
                            step=res["step"],
                            tool_name=res["tool_name"],
                            success=res["success"],
                            data=res.get("data"),
                            error=res.get("error"),
                            execution_time=res.get("execution_time"),
                            resolved_inputs=res.get("resolved_inputs")
                        )
                        for res in exec_data.get("execution_results", [])
                    ],
                    final_outputs=exec_data.get("final_outputs", {}),
                    execution_status=exec_data.get("execution_status", "unknown"),
                    total_time=exec_data.get("total_time", 0.0),
                    errors=exec_data.get("errors", [])
                )
            
            # Add metadata
            if "metadata" in result:
                meta_data = result["metadata"]
                response.metadata = WorkflowMetadata(
                    tools_used=meta_data.get("tools_used", []),
                    total_time=meta_data.get("total_time", 0.0),
                    execution_status=meta_data.get("execution_status", "unknown")
                )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Tool-first workflow processing failed: {str(e)}"
        )

@app.get("/api/workflow/capabilities")
async def get_agent_capabilities():
    """
    Get information about the Mem0-enhanced agent's capabilities and available tools.
    """
    try:
        capabilities = await tool_first_agent.explain_capabilities()
        
        return {
            "success": True,
            "capabilities": capabilities,
            "architecture": "mem0-tool-first",
            "persistent_memory": True,
            "cross_session_continuity": True,
            "memory_system": "Mem0"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get capabilities: {str(e)}"
        )

@app.get("/api/workflow/status/{request_id}")
async def get_workflow_status(request_id: str):
    """
    Get the status of a specific workflow request.
    Note: This is a placeholder for future implementation with persistent storage.
    """
    return {
        "request_id": request_id,
        "status": "completed",
        "message": "Status tracking will be implemented with database integration",
        "architecture": "tool-first"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 