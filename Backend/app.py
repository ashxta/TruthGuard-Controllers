from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, ViTImageProcessor, ViTForImageClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch
from PIL import Image
import io
import random

app = FastAPI()

# Add CORS middleware to allow cross-origin requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check for GPU
device = 0 if torch.cuda.is_available() else -1

# Load models
text_classifier = None
url_tokenizer = None
url_model = None
image_processor = None
image_model = None
try:
    # Text analysis model (changed to a more stable option)
    text_classifier = pipeline("text-classification", model="dhruvpal/fake-news-bert", device=device)
    
    # URL analysis model with manual tokenizer
    url_tokenizer = AutoTokenizer.from_pretrained("toma/distilbert-base-uncased-url-classifier")
    url_model = AutoModelForSequenceClassification.from_pretrained("toma/distilbert-base-uncased-url-classifier")
    
    # Image analysis (deepfake detection) model
    image_processor = ViTImageProcessor.from_pretrained("Wvolf/ViT_Deepfake_Detection")
    image_model = ViTForImageClassification.from_pretrained("Wvolf/ViT_Deepfake_Detection")
    
except Exception as e:
    print(f"Error loading models: {e}")

class TextAnalysisRequest(BaseModel):
    text: str

class UrlAnalysisRequest(BaseModel):
    url: str

# Helper function to format the analysis result (your required structure)
def format_result(content_type, raw_result, input_text):
    primary_result = raw_result[0] if isinstance(raw_result, list) else raw_result
    
    # Adjusting for different model outputs
    is_misinformation = primary_result.get("label", "").lower().replace(" ", "").startswith("misinformation") or primary_result.get("label", "").lower().startswith("fake") or primary_result.get("label", "").lower().startswith("defacement") or primary_result.get("label", "").lower().startswith("malware")
    is_phishing = primary_result.get("label", "").lower().startswith("phishing") or primary_result.get("label", "").lower().startswith("malicious")
    
    credibility_score = primary_result.get("score", 0.5)
    if is_misinformation or is_phishing:
        credibility_score = 1 - credibility_score

    return {
        "type": content_type,
        "credibilityScore": credibility_score,
        "analysis": f"The {content_type} has been flagged as potential {primary_result.get('label', 'misinformation')} with a confidence score of {credibility_score:.2f}.",
        "flags": {
            "potentialMisinformation": is_misinformation,
            "needsFactChecking": is_misinformation or is_phishing,
            "biasDetected": "bias" in primary_result.get("label", "").lower(),
            "manipulatedContent": is_misinformation or is_phishing,
        },
        "sources": [
            f"AI Model: {primary_result.get('label', 'N/A')}",
            "External Verification Service (Mock)",
        ],
        "details": {
            "sentiment": "Negative" if is_misinformation else "Positive/Neutral",
            "confidence": primary_result.get("score", 0.5),
            "keyTerms": input_text.split()[:5] if input_text else []
        },
    }

# Fallback function for when models fail to load
def mock_analysis(content_type, input_text):
    mock_score = random.uniform(0.3, 0.9)
    is_misinfo = mock_score < 0.6
    
    return {
        "type": content_type,
        "credibilityScore": mock_score,
        "analysis": "Analysis failed: AI models could not be loaded. This is a mock result.",
        "flags": {
            "potentialMisinformation": is_misinfo,
            "needsFactChecking": is_misinfo,
            "biasDetected": random.choice([True, False]),
            "manipulatedContent": random.choice([True, False]),
        },
        "sources": ["Offline Mode (Mock)"],
        "details": {
            "sentiment": "N/A",
            "confidence": mock_score,
            "keyTerms": input_text.split()[:5] if input_text else []
        },
    }

@app.post("/analyze/text")
async def analyze_text(request: TextAnalysisRequest):
    if not text_classifier:
        return {"result": mock_analysis("text", request.text)}
    try:
        result = text_classifier(request.text)
        formatted = format_result("text", result, request.text)
        return {"result": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@app.post("/analyze/url")
async def analyze_url(request: UrlAnalysisRequest):
    if not url_model:
        return {"result": mock_analysis("url", request.url)}
    try:
        inputs = url_tokenizer(request.url, return_tensors="pt")
        with torch.no_grad():
            outputs = url_model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        predicted_class_id = torch.argmax(probabilities).item()
        label = url_model.config.id2label[predicted_class_id]
        score = probabilities[0][predicted_class_id].item()
        
        formatted = format_result("url", [{"label": label, "score": score}], request.url)
        return {"result": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL analysis failed: {str(e)}")

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    if not image_model:
        return {"result": mock_analysis("image", file.filename)}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        inputs = image_processor(images=image, return_tensors="pt")
        outputs = image_model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()
        result = image_model.config.id2label[predicted_class_id]
        
        # Score is not directly available, but we can use softmax
        scores = torch.nn.functional.softmax(logits, dim=1)
        confidence_score = scores[0][predicted_class_id].item()

        formatted_result = format_result("image", {"label": result, "score": confidence_score}, file.filename)
        return {"result": formatted_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.post("/analyze/video")
async def analyze_video():
    raise HTTPException(status_code=501, detail="Video analysis is not yet implemented.")