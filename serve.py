"""
ETA Prediction API entrypoint.

Usage:
    python serve.py
    # Docs at http://localhost:8000/docs
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("eta_pipeline.serving.api:app", host="0.0.0.0", port=8000, reload=False)
