from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
import Modelo as modelo
import uvicorn
from PIL import Image
from fastapi.responses import JSONResponse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from fastapi.encoders import jsonable_encoder


###################### API #########################################

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to Your Recognice face FastAPI"}


@app.post("/images")
def create_upload_file(file: UploadFile = File(None)):
    
    #print(file[0])

    tipo_file = ['.jpeg', '.jpg'] 
    
    conteo_file = 0
    for j in tipo_file:
        if j not in  file.filename: 
          conteo_file = conteo_file + 1
    
    if conteo_file == len(tipo_file):
       return  HTTPException(status_code = 404, detail=  "File must have a JPEG or JPG extension")
          

    modelo_resultados = modelo.detecion_face(file.file)

    if 'My face appears in the picture' == modelo_resultados:
      return JSONResponse(status_code=200, content={"message": "My face appears in the picture"})
   
    elif 'My face does not appear in the picture' == modelo_resultados:
      return JSONResponse(status_code=200, content={"message": "My face does not appear in the picture"})

    elif 'There is any face in the pictue' == modelo_resultados:
      return JSONResponse(status_code=200, content={"message": "There is any face in the picture"})

@app.exception_handler(500)
async def internal_exception_handler(request: Request, exc: Exception):
  return JSONResponse(status_code=500, content=jsonable_encoder({"code": 500, "msg": "Internal Server Error"}))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0')
    
    
