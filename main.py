import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import time
import google.generativeai as genai
from fastapi.staticfiles import StaticFiles
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip, ColorClip, ImageClip, vfx, CompositeAudioClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json

# Patch for modern Pillow compatibility (ANTIALIAS removed in v10)
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS
from dotenv import load_dotenv

# Load environment variables from ../untitled folder/.env if needed or a dedicated backend .env
load_dotenv(dotenv_path="../untitled folder/.env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env")
else:
    genai.configure(api_key=API_KEY)

app = FastAPI(title="VEDIT AI Backend")

# Initialize directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("exports", exist_ok=True)

# Serve static files for exports
app.mount("/exports", StaticFiles(directory="exports"), name="exports")

from fastapi.responses import FileResponse

@app.get("/api/download/{filename}")
async def download_export(filename: str):
    file_path = os.path.join("exports", filename)
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path, 
            media_type="video/mp4", 
            filename=filename, 
            content_disposition_type="inline" # Allows preview, proper Cmd+S saving
        )
    raise HTTPException(status_code=404, detail="File not found")

# Enable CORS for the frontend React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini Manifest Path
UPLOAD_DIR = "uploads"
MANIFEST_PATH = os.path.join(UPLOAD_DIR, "gemini_manifest.json")

def get_manifest():
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f)

@app.post("/api/upload-media")
async def upload_media(file: UploadFile = File(...)):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="Gemini API key is not configured.")
        
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Check if file already exists locally
        if not os.path.exists(file_path):
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        # Check manifest for existing Gemini File ID
        manifest = get_manifest()
        if file.filename in manifest:
            print(f"[AI Backend] Reusing Gemini File ID for {file.filename}: {manifest[file.filename]}")
            return {
                "status": "success", 
                "geminiFileId": manifest[file.filename],
                "filename": file.filename
            }
            
        print(f"[AI Backend] Uploading {file.filename} to Gemini...")
        gemini_file = genai.upload_file(path=file_path)
        
        # Save to manifest
        manifest[file.filename] = gemini_file.name
        save_manifest(manifest)
        
        print(f"[AI Backend] Successfully uploaded to Gemini File API: {gemini_file.name}")
        
        return {
            "status": "success", 
            "geminiFileId": gemini_file.name,
            "filename": file.filename
        }
    except Exception as e:
        print(f"[AI Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    # No longer deleting file, it's needed for the export/render!

class PromptRequest(BaseModel):
    prompt: str
    clips_metadata: list

class ExportRequest(BaseModel):
    clips: list
    settings: dict = {"width": 1920, "height": 1080, "fps": 30}

def create_text_clip_fallback(text, duration, color='white', fontsize=70, size=(1920, 1080)):
    # Create a transparent RGBA image for text
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Try to find a standard font
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf"
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            font = ImageFont.truetype(path, fontsize * 2) # Higher res for better quality
            break
    if not font:
        font = ImageFont.load_default()
    
    # Bounding box calculation for centering
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Antialiased high-quality text draw
    tx, ty = (size[0] - tw) / 2, (size[1] - th) / 2
    draw.text((tx, ty), text, font=font, fill=color)
    
    # Resize back with Lanczos for smooth edges
    img = img.resize(size, Image.Resampling.LANCZOS)
    
    # Convert PIL Image to MoviePy ImageClip
    img_array = np.array(img)
    return ImageClip(img_array).set_duration(duration)

from PIL import ImageFilter
def gaussian_blur_pil(image, radius=10):
    """Robust PIL-based blur for MoviePy clips"""
    pil_img = Image.fromarray(image)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)

@app.post("/api/export")
async def export_video(request: ExportRequest):
    try:
        render_clips = []
        audio_tracks = []
        width = request.settings.get("width", 1080)
        height = request.settings.get("height", 1920)
        
        # IMPORTANT: Sort clips by z-priority 
        # Text (-2) should be on top of Video (0, 1, 2)
        def get_z_priority(c):
            ti = c.get("trackIndex", 0)
            if ti == -2: return 1000 # Text is frontmost
            if ti == -1: return 900  # Audio
            return ti                # Video 0, 1, 2... higher is front
            
        sorted_clips = sorted(request.clips, key=get_z_priority)
        
        for clip_data in sorted_clips:
            start_pos = clip_data.get("startPosition", 0) / 100.0
            duration = clip_data.get("duration", 0)
            
            # Helper to calculate dimensions from percentages
            # Canvas is 1080x1920 by default (9:16)
            clip_w_px = (clip_data.get("width", 100) / 100.0) * width
            clip_h_px = (clip_data.get("height", 100) / 100.0) * height
            clip_x_px = (clip_data.get("x", 0) / 100.0) * width
            clip_y_px = (clip_data.get("y", 0) / 100.0) * height

            if clip_data["type"] == "video":
                filename = clip_data.get("name")
                file_path = os.path.join("uploads", filename)
                if not os.path.exists(file_path): continue
                
                clip = VideoFileClip(file_path)
                
                # Trim first to avoid processing excess data
                offset = clip_data.get("videoOffset", 0)
                clip = clip.subclip(offset, min(offset + duration, clip.duration))
                
                # Resize and Position
                clip = clip.resize(newsize=(clip_w_px, clip_h_px))
                clip = clip.set_position((clip_x_px, clip_y_px))
                
                # Audio and Opacity
                clip = clip.set_opacity(clip_data.get("opacity", 100) / 100.0)
                if clip.audio and not clip_data.get("muted", False):
                    clip = clip.volumex(clip_data.get("volume", 100) / 100.0)
                else:
                    clip = clip.without_audio()
                
                # Filters (Blur)
                if "blur" in clip_data.get("cssFilter", "").lower():
                    clip = clip.fl_image(lambda img: gaussian_blur_pil(img, 15))
                
                # Timing and Transitions
                clip = clip.set_start(start_pos)
                if clip_data.get("transitionIn") == "fade":
                    clip = clip.crossfadein(0.5)
                if clip_data.get("transitionOut") == "fade":
                    clip = clip.crossfadeout(0.5)
                    
                render_clips.append(clip)
                
            elif clip_data["type"] == "image":
                filename = clip_data.get("name")
                file_path = os.path.join("uploads", filename)
                if not os.path.exists(file_path): continue
                
                img_clip = ImageClip(file_path).set_duration(duration)
                img_clip = img_clip.resize(newsize=(clip_w_px, clip_h_px))
                img_clip = img_clip.set_position((clip_x_px, clip_y_px))
                img_clip = img_clip.set_start(start_pos)
                img_clip = img_clip.set_opacity(clip_data.get("opacity", 100) / 100.0)
                
                if "blur" in clip_data.get("cssFilter", "").lower():
                    img_clip = img_clip.fl_image(lambda img: gaussian_blur_pil(img, 15))
                    
                render_clips.append(img_clip)
                
            elif clip_data["type"] == "audio":
                filename = clip_data.get("name")
                file_path = os.path.join("uploads", filename)
                if not os.path.exists(file_path): continue
                
                a_clip = AudioFileClip(file_path)
                a_clip = a_clip.subclip(0, min(duration, a_clip.duration))
                a_clip = a_clip.set_start(start_pos)
                a_clip = a_clip.volumex(clip_data.get("volume", 100) / 100.0)
                audio_tracks.append(a_clip)

            elif clip_data["type"] == "text":
                text = clip_data.get("text", "VEDIT")
                # Scale text clip based on requested width/height
                text_clip = create_text_clip_fallback(text, duration, size=(int(clip_w_px), int(clip_h_px)))
                text_clip = text_clip.set_position((clip_x_px, clip_y_px))
                text_clip = text_clip.set_start(start_pos)
                render_clips.append(text_clip)

        if not render_clips:
            raise HTTPException(status_code=400, detail="No valid clips to render.")
            
        print("[Export] Composing complex layers and audio tracks...")
        final_video = CompositeVideoClip(render_clips, size=(width, height))
        
        # Merge background audio tracks with video audio
        if audio_tracks:
            video_audio = final_video.audio
            if video_audio:
                audio_tracks.insert(0, video_audio)
            
            final_audio = CompositeAudioClip(audio_tracks)
            final_video = final_video.set_audio(final_audio)

        output_filename = f"vedit_export_{int(time.time())}.mp4"
        output_path = os.path.join("exports", output_filename)
        
        print(f"[Export] Encoding with high-fidelity audio to {output_path}...")
        # codec libx264 + aac with async 1 for better sync
        final_video.write_videofile(
            output_path, 
            fps=request.settings.get("fps", 30), 
            codec="libx264", 
            audio_codec="aac",
            audio_fps=44100,
            ffmpeg_params=["-async", "1"] 
        )
        
        # Close all to free memory
        for c in render_clips: c.close()
        for a in audio_tracks: a.close()

        return {"status": "success", "url": f"/exports/{output_filename}"}
        
    except Exception as e:
        print(f"[Export Error] {e}")
        raise HTTPException(status_code=500, detail=f"Rendering failed: {str(e)}")

@app.post("/api/analyze-prompt")
async def analyze_prompt(request: PromptRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="Gemini API key is not configured.")
        
    print(f"\n[AI Backend] Received Prompt: '{request.prompt}'")
    print(f"[AI Backend] Analyzing metadata for {len(request.clips_metadata)} clips on timeline...")
        
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        system_prompt = """
        You are an AI Video Editor Assistant with 'Vibecoding', 'Highlight Extraction', and full automation capabilities. 
        You will receive a user's natural language edit PROMPT and the current timeline CLIPS metadata.
        Output ONLY a RAW JSON array representing the NEW clips array after applying the requested edits.
        
        Capabilities:
        1. **Split/Cut & Highlights**: 
           - If user says "find highlights", "instagram reel", or "instagram short", analyze media.
           - Extract 5-10 distinct highlight segments (3-6s each).
           - RETURN each segment as a SEPARATE object in the JSON array.
        2. **Math, OFFSETS & Units (CRITICAL)**:
           - **`startPosition`** is where the clip is placed on the timeline (in Timeline Units: Seconds * 100).
           - **`videoOffset`** is VERY IMPORTANT: It is the exact start time in the original source video (in SECONDS) where this segment begins. To show different highlights, each segment MUST have a different `videoOffset`. Never leave it at 0 for all clips.
           - **`duration`** is the length of the segment in SECONDS.
           - **Alignment Rule**: For sequential videos, the `startPosition` of Clip B must be `startPosition` of Clip A + (duration of Clip A * 100).
           - **EXAMPLE PATTERN**:
             - Clip 1: startPosition: 0, videoOffset: 12.5, duration: 2.0
             - Clip 2: startPosition: 200, videoOffset: 45.0, duration: 3.5
             - Clip 3: startPosition: 550, videoOffset: 112.0, duration: 4.0
           - NEVER stack clips on the same `startPosition` unless specifically asked to overlay.
        3. **Social & Automation (TikTok/Reels)**:
           - **TikTok Style**: If asked for "tiktok", "reel", or "social style", create TWO layers for the same video:
             - Layer 1 (Background): `trackIndex: 0`, `width: 150`, `height: 150`, `x: -25`, `y: -25`, `cssFilter: 'blur(20px) brightness(0.7)'`, `muted: true`.
             - Layer 2 (Foreground): `trackIndex: 1`, `width: 80`, `height: 80`, `x: 10`, `y: 10`, `hasBorder: true`, `muted: false`.
           - ALWAYS include `transitionIn`: 'fade' and `transitionOut`: 'fade' for EVERY segment.
           - Position text overlays centrally.
        4. **Automation**: Remove silent sections, apply filters, or change volume/opacity.
        
        Rules:
        - Output the ENTIRE updated clips array.
        - **Gapless**: Align segments exactly end-to-end using `startPosition`.
        - **Variation**: Do NOT repeat the same `videoOffset`. Jump around the video to extract the best moments.
        - **Track**: Place main highlight segments on `trackIndex: 0` unless layering.
        - **Unique IDs**: Use unique IDs (e.g., clip_seg_1, clip_seg_2).
        - No markdown, no explanations, just RAW JSON array.
        """
        
        # Check if any clip has a geminiFileId
        media_files = []
        for clip in request.clips_metadata:
            if 'geminiFileId' in clip:
                try:
                    g_file = genai.get_file(clip['geminiFileId'])
                    media_files.append(g_file)
                    print(f"[AI Backend] Found and attached Gemini File: {clip['geminiFileId']}")
                    break # Usually we just need the main audio/video file for context
                except Exception as ex:
                    print(f"Warning: Could not get gemini file: {ex}")

        msg = f"PROMPT: {request.prompt}\nCLIPS: {request.clips_metadata}"
        
        contents = media_files + [system_prompt + "\n\n" + msg]
        response = model.generate_content(contents)
        
        result_text = response.text.replace("```json", "").replace("```", "").strip()
        
        import json
        new_clips = json.loads(result_text)
        
        return {"status": "success", "new_clips": new_clips}
    except Exception as e:
        print(f"Error in analyze_prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-video-suggestions")
async def analyze_video_suggestions(request: PromptRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="Gemini API key is not configured.")
        
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        system_prompt = f"""
        You are a Professional AI Video Director. 
        Analyze the current video timeline and suggest 3-5 SMART improvements.
        
        Current Timeline Data: {request.clips_metadata}
        
        Output ONLY a RAW JSON array of objects with the following structure:
        - "title": Short title (e.g., "Color Grade", "Sync Audio", "Cut Silence")
        - "description": Why this is needed.
        - "icon": A Lucide icon name (Sparkles, Zap, MicVocal, Wand2, MessageSquare)
        - "prompt": The EXACT natural language prompt to send to /api/analyze-prompt if the user clicks "Apply".
        
        Focus areas:
        1. **Audio Sync**: Align music beats to video cuts.
        2. **Visual Flow**: Consistency in transitions.
        3. **Engagement**: Highlights or text overlays.
        4. **Technical**: Normalize volume, remove silence.
        5. **Social Styles**: Suggest "TikTok Style" (Blurred background + centered overlay) for portrait videos.
        
        Example JSON output:
        [
          {{"title": "Sync Audio", "description": "Align background track to scene changes.", "icon": "Zap", "prompt": "Sync the audio on track -1 to the video cuts on track 0."}},
          {{"title": "Add Transitions", "description": "Smoother flow between clips.", "icon": "Wand2", "prompt": "Add fade transitions to all video clips."}}
        ]
        """
        
        # Attach media for context if available
        media_files = []
        for clip in request.clips_metadata:
            if 'geminiFileId' in clip:
                try:
                    g_file = genai.get_file(clip['geminiFileId'])
                    media_files.append(g_file)
                    break 
                except:
                    pass

        response = model.generate_content(media_files + [system_prompt])
        result_text = response.text.replace("```json", "").replace("```", "").strip()
        
        import json
        suggestions = json.loads(result_text)
        return {"status": "success", "suggestions": suggestions}
    except Exception as e:
        print(f"Error in suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
