import uvicorn
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting the Autonomous AI Research Agent backend server")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
