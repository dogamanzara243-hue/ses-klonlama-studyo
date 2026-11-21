import os
import tempfile
import uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import soundfile as sf

app = FastAPI(title="Ses Klonlama API (Repo 1)")

# GitHub Pages veya başka bir web arayüzünden erişim için CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = tempfile.mkdtemp(prefix="cloned_voices_")
_tts_model = None


def get_tts_model():
    global _tts_model
    if _tts_model is None:
        try:
            from f5_tts.api import F5TTS
            _tts_model = F5TTS()
        except Exception as e:
            raise RuntimeError(f"F5-TTS modeli başlatılamadı: {e}")
    return _tts_model


@app.get("/health")
def health():
    return {"status": "ok", "engine": "F5-TTS"}


@app.post("/api/clone-voice")
async def clone_voice(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    speed: float = Form(1.0),
    consent: bool = Form(False),
):
    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Lütfen kendi sesinizi veya izinli bir referans sesi kullandığınızı onaylayın.",
        )

    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Seslendirilecek metin boş olamaz.")

    if len(clean_text) > 1200:
        raise HTTPException(status_code=400, detail="Metin en fazla 1200 karakter olabilir.")

    ext = os.path.splitext(ref_audio.filename or "ref.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_ref:
        content = await ref_audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="Referans ses dosyası boş.")
        tmp_ref.write(content)
        ref_path = tmp_ref.name

    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    try:
        model = get_tts_model()
        wav, sr, _ = model.infer(
            ref_file=ref_path,
            ref_text="",
            gen_text=clean_text,
            speed=max(0.5, min(1.5, float(speed))),
        )
        sf.write(out_path, wav, sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ses klonlama hatası: {str(e)}")
    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)

    return FileResponse(
        out_path,
        media_type="audio/wav",
        filename="klonlanmis_ses.wav",
    )# app.py kodunu buraya yapıştırın
