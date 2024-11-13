from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from main import ReceiptAnalyzer
import tempfile
import os
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import base64
from config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for analyzing receipt images and extracting data",
    version=settings.PROJECT_VERSION
)

# Add security scheme
security = HTTPBasic()

# Update authentication constants
USERNAME = settings.API_USERNAME
PASSWORD = settings.API_PASSWORD

# Add authentication function specifically for secret page
def verify_secret_page_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

# Configure CORS with settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Add static files mounting
app.mount("/static", StaticFiles(directory=static_dir), name="static")

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
        # "usage": {
        #     "method": "POST",
        #     "content-type": "multipart/form-data",
        #     "required_field": "file (image file)",
        #     "supported_formats": ["image/jpeg", "image/png", "image/tiff"]
        # }
    }

@app.post("/analyze/", response_model=ReceiptResponse)
async def analyze_receipt(
    file: UploadFile = File(...)
):
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

@app.get("/analyze/tayudtsfgyuasgf", response_class=HTMLResponse)
async def test_analyze_info(credentials: HTTPBasicCredentials = Depends(verify_secret_page_credentials)):
    """
    Test endpoint that shows an HTML form for receipt analysis
    """
    return """
    <html>
        <head>
            <title>Receipt Analyzer Test</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .result-container { margin-top: 20px; }
                .image-preview { max-width: 500px; margin-top: 20px; }
                .confidence { color: #666; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <h1>Receipt Analyzer Test</h1>
            <form action="/analyze/tayudtsfgyuasgf" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Analyze Receipt</button>
            </form>
            <div id="result"></div>
        </body>
    </html>
    """

@app.post("/analyze/tayudtsfgyuasgf", response_class=HTMLResponse)
async def test_analyze_receipt(
    file: UploadFile = File(...),
    credentials: HTTPBasicCredentials = Depends(verify_secret_page_credentials)
):
    """
    Test endpoint that shows analysis results in HTML format
    """
    try:
        # Read the image file
        contents = await file.read()
        
        # Save to temp file and analyze
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        # Analyze receipt
        result = analyzer.analyze_receipt(temp_path)
        
        # Convert image to base64 for display
        image_base64 = base64.b64encode(contents).decode()
        
        # Create HTML response
        html_content = f"""
        <html>
            <head>
                <title>Receipt Analysis Results</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .result-container {{ 
                        display: flex; 
                        gap: 40px; 
                        align-items: start;
                    }}
                    .results {{ 
                        flex: 1;
                        background: #f5f5f5;
                        padding: 20px;
                        border-radius: 8px;
                    }}
                    .image-container {{ 
                        flex: 1;
                        max-width: 500px;
                    }}
                    img {{ max-width: 100%; }}
                    .confidence {{ 
                        color: {'green' if result['confidence_score'] > 0.7 else 'orange'};
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <h1>Receipt Analysis Results</h1>
                <div class="result-container">
                    <div class="results">
                        <h2>Extracted Data:</h2>
                        <p><strong>Merchant:</strong> {result['merchant_name'] or 'Not found'}</p>
                        <p><strong>Date:</strong> {result['bill_date'] or 'Not found'}</p>
                        <p><strong>Amount:</strong> {result['amount']} {result['currency'] or ''}</p>
                        <p><strong>Type:</strong> {result['type']}</p>
                        <p><strong>Description:</strong> {result['description'] or 'Not found'}</p>
                        <p class="confidence">Confidence Score: {result['confidence_score']:.2f}</p>
                    </div>
                    <div class="image-container">
                        <h2>Uploaded Receipt:</h2>
                        <img src="data:image/{file.content_type};base64,{image_base64}" alt="Receipt">
                    </div>
                </div>
                <p><a href="/analyze/tayudtsfgyuasgf">⬅ Analyze Another Receipt</a></p>
            </body>
        </html>
        """
        
        # Cleanup
        os.unlink(temp_path)
        
        return html_content
        
    except Exception as e:
        error_html = f"""
        <html>
            <head>
                <title>Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .error {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Error Processing Receipt</h1>
                <p class="error">{str(e)}</p>
                <p><a href="/analyze/tayudtsfgyuasgf">⬅ Try Again</a></p>
            </body>
        </html>
        """
        return error_html

if __name__ == "__main__":
    uvicorn.run(
        "api:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT,
        reload=False,  # Disable reload in production
        # workers=4  # Number of worker processes
    )
