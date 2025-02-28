from app.main import app

# This makes the app available at the module level for gunicorn
from fastapi import FastAPI

# Re-export the app instance
application = app

# For compatibility with both gunicorn and uvicorn
__all__ = ['app', 'application']
