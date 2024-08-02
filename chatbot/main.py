from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from router import router


app = FastAPI(title="ai_toolkit", description="ai tookit for payever.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify the correct origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=600)
