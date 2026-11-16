
import io
import base64
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image

app = FastAPI(title="Khyber AI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "stabilityai/sd-turbo"
pipeline = None


class ImageRequest(BaseModel):
    prompt: str


@app.get("/")
def home():
    return {"message": "Khyber AI Studio Image Engine"}


@app.post("/generate")
def generate_image(request: ImageRequest):
    global pipeline

    if not request.prompt.strip():
        raise HTTPException(400, "Please enter an image prompt.")

    try:
        if pipeline is None:
            pipeline = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                use_safetensors=True
            )
            pipeline.to("cpu")

        image = pipeline(
            prompt=request.prompt,
            num_inference_steps=1,
            guidance_scale=0.0
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return {"image": encoded}

    except Exception as error:
        raise HTTPException(500, str(error))
