from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
from src.model.blip2_inference import BLIP2Pipeline

app = FastAPI()
pipeline = BLIP2Pipeline()

@app.post("/caption")
async def generate_caption(image: UploadFile = File(...)):
    contents = await image.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    caption = pipeline.caption_image(image_pil)
    return {"caption": caption}

@app.post("/vqa")
async def visual_question_answering(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    contents = await image.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    answer = pipeline.vqa(image_pil, question)
    return {"answer": answer}
