from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from main import ReceiptAnalyzer
import tempfile
import os

app = FastAPI(
    title="Receipt Analyzer API",
    description="API for analyzing receipt images and extracting data",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize receipt analyzer
analyzer = ReceiptAnalyzer()

class ReceiptResponse(BaseModel):
    merchant_name: Optional[str]
    bill_date: Optional[str]
    amount: Optional[str]
    currency: Optional[str]
    description: Optional[str]
    type: Optional[str]
    confidence_score: float

@app.get("/analyze/")
async def analyze_receipt_info():
    """
    Provide information about how to use the analyze endpoint
    """
    return {
        "message": "This endpoint accepts POST requests only",
        "usage": {
            "method": "POST",
            "content-type": "multipart/form-data",
            "required_field": "file (image file)",
            "supported_formats": ["image/jpeg", "image/png", "image/tiff"]
        }
    }

@app.post("/analyze/", response_model=ReceiptResponse)
async def analyze_receipt(file: UploadFile = File(...)):
    """
    Analyze a receipt image and extract information
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Create temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        # Analyze the receipt
        result = analyzer.analyze_receipt(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return ReceiptResponse(**result)
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error processing receipt: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
