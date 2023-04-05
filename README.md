# ¿Comó detectar tu rostro en una imagen?

En este proyecto se utilizan modelos de reconocimiento facial para detectar tu rostro en una imagen. La implementación se realizó utilizando Docker.

Para detectar el rostro en una imagen, primero se utilizo un modelo de detección facial llamado Multi-Task Cascaded Convolutional Neural Networks (MTCNN), que es capaz de detectar múltiples rostros en una imagen.

Una vez detectado el rostro en la imagen, se utiliza el modelo VGGFace para generar un embedding (caracterización) del rostro. El modelo VGGFace es una red neuronal convolucional profunda diseñada específicamente para tareas de reconocimiento facial y ha sido entrenado con millones de imágenes de rostros. El embedding generado se comparó con el embedding del rostro que se quiere detectar, utilizando la distancia coseno como métrica de similitud.

Como se menciono anteriormente, la implementación del modelo se realizó utilizando FastAPI y Docker. 

