import os
import sys
import torch
import uvicorn
import uuid
import gc
import traceback
import trimesh
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import requests
import base64

# --- OPTIMISATION MÉMOIRE CUDA ---
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Autoriser PyTorch à utiliser toute la mémoire disponible avec gestion intelligente
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# --- VERSION DU SERVEUR ---
SERVER_VERSION = "v15.1_STACK_2026_FINAL"

# Diagnostic au boot
print(f"🚀 TRELLIS BOOT - {SERVER_VERSION}")
print(f"🔹 Torch: {torch.__version__} | CUDA: {torch.version.cuda}")
print(f"🔹 GPU: {torch.cuda.get_device_name(0)}")

# Afficher les stats mémoire
total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"🔹 Mémoire GPU: {total_mem:.2f} GB")

app = FastAPI(title=f"Trellis API {SERVER_VERSION}")

# Configuration CORS pour OpenWebUI
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permettre tous les origins pour OpenWebUI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "/app/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

# Import Trellis (Après installation des wheels)
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# Configuration des GPUs
NUM_GPUS = torch.cuda.device_count()
print(f"🔹 GPUs détectés: {NUM_GPUS}")
for i in range(NUM_GPUS):
    gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({gpu_mem:.2f}GB)")

# Chargement du modèle LARGE (optimal avec 2 GPUs)
MODEL_NAME = "JeffreyXiang/TRELLIS-image-large"
print(f"⏳ Chargement du modèle LARGE...")

print("💾 Optimisation: activation de la réduction de mémoire...")
torch.set_float32_matmul_precision('medium')

pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_NAME)

# Distribution intelligente des modèles sur les 2 GPUs
if NUM_GPUS >= 2:
    print(f"🚀 Distribution intelligente des modèles sur 2 GPUs...")
    
    # GPU:0 - Modèles de diffusion/flow (lourds)
    # gpu0_models = ['sparse_structure_flow_model', 'slat_flow_model']
    gpu0_models = ['sparse_structure_flow_model']
    for model_name in gpu0_models:
        if model_name in pipeline.models:
            pipeline.models[model_name].to("cuda:0")
            print(f"  📌 {model_name} -> GPU:0")
    
    # GPU:1 - Modèles d'encodage/décodage (légers)
    gpu1_models = [
        'slat_flow_model',
        'image_cond_model',
        'sparse_structure_decoder',
        'slat_decoder_gs',
        'slat_decoder_rf',
        'slat_decoder_mesh'
    ]
    for model_name in gpu1_models:
        if model_name in pipeline.models:
            pipeline.models[model_name].to("cuda:1")
            print(f"  📌 {model_name} -> GPU:1")
    
    # Lister les modèles chargés
    print("\n  📊 Modèles disponibles dans le pipeline:")
    for key in pipeline.models:
        try:
            device = next(pipeline.models[key].parameters()).device
            print(f"     - {key}: {device}")
        except:
            print(f"     - {key}: unknown device")
else:
    print(f"🚀 Chargement du modèle sur GPU:0...")
    pipeline.to("cuda:0")

# Forcer la conversion en half precision (FP16) si disponible
try:
    for key in pipeline.models:
        model = pipeline.models[key]
        if hasattr(model, 'half'):
            model.half()
    print("✓ Modèles convertis en FP16 (float16)")
except Exception as e:
    print(f"⚠️ Conversion FP16 partielle: {str(e)}")

torch.cuda.empty_cache()

class GenRequest(BaseModel):
    image_url: str = None
    image_base64: str = None
    seed: int = 42
    steps: int = 12  # 12 steps avec 2 GPUs - bonne qualité
    server_host: str = "http://0.0.0.0:5000/"  # URL de base pour les fichiers

def download_image_from_url(url: str) -> Image.Image:
    """Télécharger une image depuis une URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Erreur téléchargement image: {str(e)}")

def decode_image_from_base64(b64_str: str) -> Image.Image:
    """Décoder une image depuis une chaîne base64"""
    try:
        # Supprimer les espaces et les sauts de ligne
        b64_str = b64_str.strip()
        
        # Gérer le cas où la chaîne commence par "data:image/...;base64,"
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        
        # Ajouter le padding si nécessaire
        missing_padding = len(b64_str) % 4
        if missing_padding:
            b64_str += '=' * (4 - missing_padding)
        
        image_bytes = base64.b64decode(b64_str)
        image = Image.open(BytesIO(image_bytes))
        print(f"✓ Image décodée: {image.size} {image.mode}")
        return image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Erreur décodage image base64: {str(e)}")


@app.get("/health")
async def health():
    """Endpoint de santé pour OpenWebUI"""
    return {
        "status": "ok",
        "version": SERVER_VERSION,
        "gpu_count": NUM_GPUS,
        "output_dir": OUTPUT_DIR
    }


@app.get("/viewer", response_class=HTMLResponse)
async def viewer(file: str = None):
    """
    Page de visualisation GLB avec model-viewer
    Usage: /viewer?file=nom_du_fichier.glb
    """
    if not file:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trellis 3D Viewer</title>
            <style>
                body { margin: 0; padding: 20px; background: #1a1a1a; color: white; font-family: Arial; }
                .container { text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Trellis 3D Viewer</h1>
                <p>Utilisez: /viewer?file=nom_du_fichier.glb</p>
            </div>
        </body>
        </html>
        """
    
    # Vérifier que le fichier existe
    file_path = os.path.join(OUTPUT_DIR, file)
    if not os.path.exists(file_path):
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Erreur</title>
            <style>
                body {{ margin: 0; padding: 20px; background: #1a1a1a; color: red; font-family: Arial; }}
            </style>
        </head>
        <body>
            <h1>❌ Fichier non trouvé: {file}</h1>
        </body>
        </html>
        """
    
    # Retourner la page HTML avec model-viewer
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trellis 3D Viewer - {file}</title>
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #1a1a1a; font-family: Arial, sans-serif; }}
            #container {{ width: 100vw; height: 100vh; display: flex; flex-direction: column; }}
            model-viewer {{ flex: 1; }}
            .info {{ background: #2a2a2a; color: white; padding: 10px; text-align: center; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div id="container">
            <model-viewer 
                id="viewer" 
                src="/files/{file}" 
                camera-controls 
                auto-rotate 
                ar 
                style="width: 100%; height: 100%;"
            ></model-viewer>
            <div class="info">📄 {file}</div>
        </div>
    </body>
    </html>
    """


@app.post("/generate")
async def generate(request: GenRequest, http_request: Request):
    try:
        # Déterminer le vrai host depuis les en-têtes HTTP
        server_url = str(http_request.base_url).rstrip('/')
        # Nettoyage agressif de la mémoire
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Afficher l'état de la mémoire
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"💾 Mémoire GPU avant: {allocated:.2f}GB allocée, {reserved:.2f}GB réservée")

        # Déterminer la source de l'image
        image = None
        
        # Si image_url contient du base64
        if request.image_url:
            print(f"📥 Analyse de image_url (longueur: {len(request.image_url)})...")
            
            # Vérifier si c'est du base64 (commence par data: ou est très long)
            if request.image_url.startswith("data:") or len(request.image_url) > 1000 and "," in request.image_url:
                print(f"🔍 Détecté: base64 encoding")
                image = decode_image_from_base64(request.image_url)
            else:
                # C'est une URL normale
                print(f"🔍 Détecté: URL normale")
                image = download_image_from_url(request.image_url)
        
        elif request.image_base64:
            print(f"📥 Décodage de l'image depuis image_base64...")
            image = decode_image_from_base64(request.image_base64)
        
        if not image:
            raise ValueError("Vous devez fournir 'image_url' ou 'image_base64'")
        
        # Génération
        print(f"⚙️ Génération en cours (Seed {request.seed}, Steps {request.steps})...")
        with torch.no_grad():
            outputs = pipeline.run(
                image,
                seed=request.seed,
                sparse_structure_sampler_params={"steps": request.steps},
                slat_sampler_params={"steps": request.steps}
            )

        file_id = str(uuid.uuid4())
        path = os.path.join(OUTPUT_DIR, f"{file_id}.stl")

        # Extraction et traitement du Mesh
        print(f"🎨 Traitement du mesh...")
        mesh_result = outputs['mesh'][0]
        
        # Convertir les vertices en float32 si nécessaire (VTK ne supporte pas float16)
        vertices = mesh_result.vertices
        if vertices.dtype == torch.float16:
            vertices = vertices.float()
        vertices_np = vertices.detach().cpu().numpy()
        faces_np = mesh_result.faces.detach().cpu().numpy()
        
        # Créer directement un trimesh depuis le résultat
        print(f"💾 Sauvegarde du modèle STL...")
        #trimesh_obj = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
        
        # Essayer postprocessing si GS est disponible
        #if outputs.get('gs', None) is not None:
        #    try:
        #        trimesh_obj_pp = postprocessing_utils.to_glb(
        #            app_rep=outputs['gs'],
        #            mesh=mesh_result,
        #            simplify=0.95,
        #            fill_holes=True,
        #            texture_size=1024,
        #            verbose=False
        #        )
        #        trimesh_obj = trimesh_obj_pp
        #        print(f"✅ Postprocessing appliqué")
        #    except Exception as e:
        #        print(f"⚠️ Postprocessing échoué, utilisation du mesh brut: {str(e)}")
        try:
            trimesh_obj = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,
                fill_holes=True,
                texture_size=1024
            )
            print(f"✅ Postprocessing appliqué")
        except Exception as e:
            print(f"⚠️ Postprocessing échoué, utilisation du mesh brut: {str(e)}")
            # Fallback mesh brut (sera blanc)
            vertices = outputs['mesh'][0].vertices.float().detach().cpu().numpy()
            faces = outputs['mesh'][0].faces.detach().cpu().numpy()
            trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        import numpy as np
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.radians(-90), [1, 0, 0]
        )
        trimesh_obj.apply_transform(rotation_matrix)
        print(f"🔄 Orientation corrigée (-90° X)")
        
        # Exporter en GLB
        trimesh_obj.export(path, file_type='stl')

        # Vérifier que le fichier existe
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024 * 1024)  # En MB
            print(f"✅ Modèle sauvegardé: {path} ({file_size:.2f} MB)")
            
            # Créer l'URL absolue pour OpenWebUI using the request's base URL
            file_url = f"/files/{file_id}.stl"
            print(f"📥 URL pour OpenWebUI: {file_url}")
            
            return {
                "status": "success", 
                "download_url": file_url,
                "file_id": file_id,
                "local_path": path
            }
        else:
            raise Exception(f"Fichier non créé: {path}")
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "detail": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
