from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
import Model as model
import uvicorn
from PIL import Image
from fastapi.responses import JSONResponse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from fastapi.encoders import jsonable_encoder

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to Your Recognice face FastAPI"}


@app.post("/images")
def create_upload_file(file: UploadFile = File(None)):
    
    #print(file[0])

    type_file = ['.jpeg', '.jpg'] 
    
    count_file = 0
    for j in type_file:
        if j not in  file.filename: 
          count_file = count_file + 1
    
    if count_file == len(type_file):
       return  HTTPException(status_code = 404, detail=  "File must have a JPEG or JPG extension")
          

    modelo_resultados = model.face_detection(file.file)

    if 'My face appears in the picture' == modelo_resultados:
      return JSONResponse(status_code=200, content={"message": "My face appears in the picture"})
   
    elif 'My face does not appear in the picture' == modelo_resultados:
      return JSONResponse(status_code=200, content={"message": "My face does not appear in the picture"})

    elif 'The picture does not contain any faces' == modelo_resultados:
      return JSONResponse(status_code=200, content={"message": 'The picture does not contain any faces'})


@app.exception_handler(500)
async def internal_exception_handler(request: Request, exc: Exception):
  return JSONResponse(status_code=500, content=jsonable_encoder({"code": 500, "msg": "Internal Server Error"}))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0')
    
    
