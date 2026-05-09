# api.py
# Objetivo: exponer el sistema de identificación como una API HTTP
# que el frontend puede consumir

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import json
import numpy as np
import faiss
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── 1. Inicializar FastAPI ────────────────────────────────────────────────────

app = FastAPI(
    title="NosePrint ID",
    description="API de identificación de perros por huella nasal",
    version="1.0.0"
)

# CORS: permite que el frontend (corriendo en otro puerto) consuma la API
# Sin esto el navegador bloquearía las peticiones
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",                        # desarrollo local
        "https://noseprint-frontend.vercel.app",        # producción Vercel
    ],   # en producción especificarías el dominio exacto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. Cargar el modelo al iniciar la API ─────────────────────────────────────
#
# Hacemos esto UNA sola vez al arrancar.
# Si lo hiciéramos en cada petición, cada identificación tardaría ~3 segundos
# solo en cargar el modelo.

#print("Cargando modelo ResNet-50...")
# ANTES Cambiando ResNet-50 por MobileNetV3
#modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#modelo.fc = torch.nn.Identity()
#modelo.eval()
# DESPUÉS
print("Cargando modelo MobileNetV3...")
modelo = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
modelo.classifier = torch.nn.Identity()
modelo.eval()
print("✓ Modelo listo")

transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── 3. Cargar el índice FAISS y metadata ──────────────────────────────────────

print("Cargando índice FAISS...")
indice   = faiss.read_index("database/indice.faiss")
metadata = json.load(open("database/metadata.json", encoding="utf-8"))
perros   = json.load(open("database/perros.json",   encoding="utf-8"))
print(f"✓ Índice listo — {indice.ntotal} vectores indexados")

# ── 4. Funciones auxiliares ───────────────────────────────────────────────────

def imagen_a_embedding(bytes_imagen: bytes) -> np.ndarray:
    """
    Recibe los bytes de una imagen (lo que llega por HTTP)
    y retorna su embedding normalizado como array numpy.
    """
    imagen = Image.open(io.BytesIO(bytes_imagen)).convert("RGB")
    tensor = transformacion(imagen)
    batch  = tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = modelo(batch)

    embedding_np = embedding.squeeze(0).numpy()

    # Normalizar a longitud 1
    norma = np.linalg.norm(embedding_np)
    if norma > 0:
        embedding_np = embedding_np / norma

    return embedding_np.astype(np.float32)


def buscar_en_indice(embedding: np.ndarray, k: int = 5):
    consulta = embedding.reshape(1, -1)
    scores, indices = indice.search(consulta, k)

    #print(f"\nDEBUG buscar_en_indice:")
    #print(f"  Top {k} resultados crudos de FAISS:")

    votos = {}
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        meta     = metadata[idx]
        perro_id = meta["perro_id"]
        #print(f"    idx={idx} perro={perro_id} score={score:.4f}")
        if perro_id not in votos:
            votos[perro_id] = []
        votos[perro_id].append(float(score))

    #print(f"  Votos agrupados:") PARA DEBUGEAR
    #for pid, ss in votos.items():
    #    print(f"    {pid}: votos={len(ss)} max={max(ss):.4f} mean={np.mean(ss):.4f}")

    if not votos:
        return None, 0.0, {}

    # ANTES — prioriza votos, desempata por score
    #ganador   = max(votos, key=lambda p: (len(votos[p]), max(votos[p])))
    # AHORA — prioriza score máximo directamente
    ganador = max(votos, key=lambda p: max(votos[p]))
    confianza = float(max(votos[ganador]))

    #print(f"  Ganador: {ganador}, confianza: {confianza:.4f}")

    scores_por_perro = {
        pid: float(max(s)) for pid, s in votos.items()
    }

    return ganador, confianza, scores_por_perro

# ── 5. Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def raiz():
    """Endpoint de verificación — confirma que la API está corriendo."""
    return {
        "estado"  : "activo",
        "version" : "1.0.0",
        "perros_registrados": len(perros),
        "vectores_indexados": indice.ntotal,
    }


@app.get("/perros")
def listar_perros():
    """Retorna todos los perros registrados en la base de datos."""
    resultado = []
    for perro_id, info in perros.items():
        resultado.append({
            "id"     : perro_id,
            "nombre" : info["nombre"],
            "raza"   : info["raza"],
            "dueno"  : info["dueno"],
            "edad"   : info["edad"],
        })
    return {"perros": resultado, "total": len(resultado)}


@app.post("/identificar")
async def identificar(foto: UploadFile = File(...)):
    """
    Recibe una foto de nariz de perro e identifica al perro.

    Retorna:
    - encontrado     : si se identificó al perro
    - perro          : datos del perro identificado
    - confianza      : score de similitud del ganador (0-1)
    - scores         : score de cada perro candidato
    """
    # Validar que es una imagen
    if not foto.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (jpg, png, etc.)"
        )

    # Leer los bytes de la imagen
    bytes_imagen = await foto.read()

    # Extraer embedding
    try:
        embedding = imagen_a_embedding(bytes_imagen)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar la imagen: {str(e)}"
        )

    # Buscar en el índice
    ganador_id, confianza, scores = buscar_en_indice(embedding, k=5)

    if ganador_id is None:
        return {"encontrado": False, "mensaje": "Base de datos vacía"}

    # Umbral mínimo de confianza
    # Si el score es muy bajo, probablemente el perro no está registrado
    UMBRAL = 0.85
    encontrado = confianza >= UMBRAL

    # Agrega esto justo antes de "respuesta = {"
    #print(f"DEBUG — ganador: {ganador_id}, confianza: {confianza}, scores: {scores}")

    respuesta = {
        "encontrado" : encontrado,
        "confianza"  : round(confianza, 4),
        "scores"     : {pid: round(s, 4) for pid, s in scores.items()},
    }

    if encontrado and ganador_id in perros:
        info = perros[ganador_id]
        respuesta["perro"] = {
            "id"     : ganador_id,
            "nombre" : info["nombre"],
            "raza"   : info["raza"],
            "dueno"  : info["dueno"],
            "edad"   : info["edad"],
        }
    elif not encontrado:
        respuesta["mensaje"] = (
            f"Confianza insuficiente ({confianza:.2f} < {UMBRAL}). "
            f"El perro probablemente no está registrado."
        )

    return respuesta


@app.post("/registrar")
async def registrar(
    foto    : UploadFile = File(...),
    nombre  : str = Form(...),
    raza    : str = Form("Desconocida"),
    dueno   : str = Form("Sin registrar"),
    edad    : int = Form(0),
):
    """
    Registra un nuevo perro en la base de datos.

    Recibe la foto de la nariz + datos del perro.
    Extrae el embedding y lo agrega al índice FAISS.
    """
    global indice, metadata, perros

    # Validar imagen
    if not foto.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Debe ser una imagen")

    bytes_imagen = await foto.read()

    # Extraer embedding
    try:
        embedding = imagen_a_embedding(bytes_imagen)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar imagen: {str(e)}"
        )

    # Generar nuevo ID
    ids_existentes = [int(pid.split("_")[1]) for pid in perros.keys()]
    nuevo_numero   = max(ids_existentes) + 1 if ids_existentes else 1
    nuevo_id       = f"perro_{nuevo_numero:03d}"

    # Guardar embedding en disco
    os.makedirs(f"data/train/{nuevo_id}", exist_ok=True)
    ruta_npy = f"data/train/{nuevo_id}/foto_01.npy"
    np.save(ruta_npy, embedding)

    # Agregar al índice FAISS
    indice.add(embedding.reshape(1, -1))

    # Agregar a metadata
    metadata.append({
        "perro_id" : nuevo_id,
        "foto"     : "foto_01.jpg",
        "ruta"     : ruta_npy,
    })

    # Agregar a perros
    perros[nuevo_id] = {
        "nombre" : nombre,
        "raza"   : raza,
        "dueno"  : dueno,
        "edad"   : edad,
    }

    # Persistir cambios en disco
    faiss.write_index(indice, "database/indice.faiss")
    with open("database/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open("database/perros.json", "w", encoding="utf-8") as f:
        json.dump(perros, f, indent=2, ensure_ascii=False)

    return {
        "registrado" : True,
        "id"         : nuevo_id,
        "nombre"     : nombre,
        "mensaje"    : f"{nombre} registrado exitosamente con ID {nuevo_id}",
    }
    
# Punto de entrada para Railway
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)