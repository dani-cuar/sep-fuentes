# import os
# import shutil

# def clasificar_archivos_directorio(ruta_directorio, ruta_salida_pares, ruta_salida_impares):
#     """
#     Esta función recorre todos los archivos en un directorio y clasifica los archivos en dos carpetas distintas,
#     basándose en si su índice (basado en el orden alfabético) es par o impar.

#     :param ruta_directorio: Ruta del directorio que contiene los archivos a clasificar.
#     :param ruta_salida_pares: Ruta del directorio de salida para los archivos con índice par.
#     :param ruta_salida_impares: Ruta del directorio de salida para los archivos con índice impar.
#     """
    
#     # Crear los directorios de salida si no existen
#     os.makedirs(ruta_salida_pares, exist_ok=True)
#     os.makedirs(ruta_salida_impares, exist_ok=True)
    
#     # Obtener la lista de todos los archivos en el directorio especificado
#     archivos = [archivo for archivo in os.listdir(ruta_directorio) if os.path.isfile(os.path.join(ruta_directorio, archivo))]
#     archivos.sort()  # Asegurar un orden alfabético
    
#     # Clasificar cada archivo como par o impar y moverlo a la carpeta correspondiente
#     for indice, archivo in enumerate(archivos, start=1):  # Comenzar la enumeración en 1 para el primer archivo
#         ruta_completa_archivo = os.path.join(ruta_directorio, archivo)
        
#         # Determinar si el índice es par o impar
#         if indice % 2 == 0:  # Índice par
#             shutil.move(ruta_completa_archivo, os.path.join(ruta_salida_pares, archivo))
#         else:  # Índice impar
#             shutil.move(ruta_completa_archivo, os.path.join(ruta_salida_impares, archivo))
            
# # Ejemplo de cómo llamar a la función
# # Suponiendo que tienes un directorio 'mis_archivos' y quieres mover los archivos a 'archivos_pares' y 'archivos_impares'
# clasificar_archivos_directorio('C:/Users/Daniela Cuartas/Desktop/recortes', 'C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/audios separados/y2', 'C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/audios separados/y1')

# Nota: Reemplaza 'ruta/a/mis_archivos', 'ruta/a/archivos_pares' y 'ruta/a/archivos_impares' con tus rutas reales.

import torch
from torchvision import models
from torchsummary import torchsummary

# Descarga el modelo VGG19 pre-entrenado
model_vgg19 = models.vgg19(pretrained=True)

# Mueve el modelo a la GPU si está disponible, de lo contrario, se queda en la CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_vgg19 = model_vgg19.to(device)

# Imprime el resumen del modelo
# Asume que la entrada es de tamaño (3, 224, 224) como es típico para modelos VGG
print(torchsummary.summary(model_vgg19, (3, 224, 224)))

