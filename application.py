import os
import io
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re

app = FastAPI()
app.add_middleware(HTTPSRedirectMiddleware)

UPLOAD_DIR = "uploads"
SIGNATURE_DIR = "signatures"
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SIGNATURE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/signatures", StaticFiles(directory=SIGNATURE_DIR), name="signatures")
templates = Jinja2Templates(directory="templates")

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def extract_signature(image):
    preprocessed = preprocess_image(image)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours[:10]:  # Check the 10 largest contours
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        
        # Adjust these criteria based on your typical signature size and shape
        if 1000 < area < 50000 and 1 < aspect_ratio < 7:
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.width - x, w + 2*padding)
            h = min(image.height - y, h + 2*padding)
            
            signature = image.crop((x, y, x+w, y+h))
            return signature, (x, y, w, h)
    
    return None, None

def extract_name(image, signature_box):
    if signature_box:
        x, y, w, h = signature_box
        # Look for text above the signature
        text_region = image.crop((x, max(0, y-50), x+w, y))
        local_text = pytesseract.image_to_string(text_region)
        
        # Look for a name-like pattern (e.g., two or three capitalized words)
        name_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', local_text)
        if name_match:
            return name_match.group(0)
    
    # If no name found near signature, search the entire image
    text = pytesseract.image_to_string(image)
    name_match = re.search(r'name\s+of\s+signature[:\s]*([^\n]+)', text, re.IGNORECASE)
    if name_match:
        return name_match.group(1).strip()
    
    return "Name not found"

def process_cheque(file_path):
    doc = fitz.open(file_path)
    if len(doc) != 1:
        raise ValueError("Expected a single-page PDF")
    
    page = doc[0]
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    signature, signature_box = extract_signature(img)
    if signature:
        signature_filename = f"signature_{uuid.uuid4()}.png"
        signature_path = os.path.join(SIGNATURE_DIR, signature_filename)
        signature.save(signature_path)
        
        name = extract_name(img, signature_box)
        
        return {
            "name": name,
            "signature_image": f"/signatures/{signature_filename}"
        }
    else:
        return {"error": "No signature found"}

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.pdf")
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        result = process_cheque(file_path)
        os.remove(file_path)
        return JSONResponse(content=result)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signature/{filename}")
async def get_signature(filename: str):
    file_path = os.path.join(SIGNATURE_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Signature image not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)