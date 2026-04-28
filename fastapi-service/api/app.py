import sys
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import io
import base64
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from services.enhance_service import EnhanceService



LLE_CHECKPOINT = os.path.join(BASE_DIR, "models", "alle_best_model.pt")

_service: EnhanceService = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service

    print(" Loading LLE model...")
    _service = EnhanceService(lle_checkpoint=LLE_CHECKPOINT)
    print(" Model ready")

    yield
    _service = None



app = FastAPI(title="ALLE Enhancement API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
async def health():
    return {"status": "ok"}



@app.post("/api/enhance")
async def enhance(file: UploadFile = File(...)):
    try:
        
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Upload an image file")

        
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")

        
        lle_buf, denoise_buf = _service.enhance_to_bytes(pil)

        
        original_buf = io.BytesIO()
        pil.save(original_buf, format="PNG")
        original_buf.seek(0)

        
        response = {
            "original_image": base64.b64encode(original_buf.getvalue()).decode(),
            "lle_image": base64.b64encode(lle_buf.getvalue()).decode(),
            "denoised_image": base64.b64encode(denoise_buf.getvalue()).decode(),
        }

        return response

    except Exception as e:
        
        print("\nERROR OCCURRED:")
        traceback.print_exc()

        
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )