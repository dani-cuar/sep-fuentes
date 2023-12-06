import glob
import os
from pydub import AudioSegment
import random
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

def listar_archivos_en_carpeta(ruta):
    # Utiliza el patrón '**/*' para recorrer todas las subcarpetas y archivos
    archivos = glob.glob(os.path.join(ruta, '**/*.wav'), recursive=True)
    # Filtra solo los archivos, excluyendo los directorios
    archivos = [archivo for archivo in archivos if os.path.isfile(archivo)]
    
    return archivos

def separar_archivos_por_genero(archivos_wav):
    archivos_male = [archivo for archivo in archivos_wav if "RoroaMale" in archivo]
    archivos_female = [archivo for archivo in archivos_wav if "RoroaFemale" in archivo]

    return archivos_male, archivos_female

def recortar_archivo(path):
    duracion_final = 90 * 1000  # Duración final en milisegundos (1 minuto y 30 segundos)
    audio = AudioSegment.from_file(path)

    # Seleccionar aleatoriamente entre 0 y 30 seg
    punto_inicio = random.choice([0, len(audio) - duracion_final])

    # Recortar el audio desde el punto de inicio hasta la duración final
    audio_recortado = audio[punto_inicio:punto_inicio + duracion_final] #0-1:30 / 30-2:00

    # Obtener el nombre del archivo original
    nombre_original = os.path.basename(path)

    carpeta_destino = "C:/Users/Daniela Cuartas/OneDrive - Universidad de Antioquia/Tesis/recortes"

    # Construir la ruta para el archivo recortado en la carpeta destino
    ruta_destino = os.path.join(carpeta_destino, nombre_original)

    # Guardar el archivo de audio recortado
    audio_recortado.export(ruta_destino, format="wav")

    return ruta_destino

def sumar_audios(archivos_male, archivos_female, resultado_folder):
    for audio_male_path, audio_female_path in zip(archivos_male, archivos_female):
        # Recortar los archivos antes de cargarlos
        audio_male_recortado = recortar_archivo(audio_male_path) # devuelve la ruta del archivo recortado
        audio_female_recortado = recortar_archivo(audio_female_path)
        # Cargar ambos archivos de audio
        audio_male, sr1 = librosa.load(audio_male_recortado, sr=None)  
        audio_female, sr2 = librosa.load(audio_female_recortado, sr=None) 

        # Resamplear si es necesario
        if sr1 > sr2:
            audio_male = librosa.resample(y=audio_male, orig_sr=sr1, target_sr=sr2)
            sr1 = sr2
        elif sr2 > sr1:
            audio_female = librosa.resample(y=audio_female, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1

        # Ajustar la longitud
        min_length = min(len(audio_male), len(audio_female))
        audio_male = audio_male[:min_length]
        audio_female = audio_female[:min_length]

        # Sumar los audios
        audio_sumado = audio_male + audio_female

        # Extraer los nombres base sin la extensión y formar el nombre del archivo resultante
        base1 = os.path.basename(audio_male_path).split('.')[0]
        base2 = os.path.basename(audio_female_path).split('.')[0]
        resultado_name = f"{base1}_{base2}.wav"
        resultado_path = os.path.join(resultado_folder, resultado_name)

        # Guardar el archivo de audio sumado
        sf.write(resultado_path, audio_sumado, sr1)

        # Calcular el espectrograma
        espectrograma = librosa.amplitude_to_db(np.abs(librosa.stft(audio_sumado)), ref=np.max)

        # Construir la ruta para guardar el espectrograma como imagen
        espectrograma_img_path = os.path.join(resultado_folder, f"{base1}_{base2}_espectrograma.png")

        # Construir el título con los nombres de los archivos originales
        titulo_espectrograma = f'Espectrograma de {base1} y {base2}'

        # Guardar el espectrograma como imagen
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(espectrograma, sr=sr1, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        #plt.title(titulo_espectrograma)
        plt.title("Espectrograma del audio sumado")
        plt.tight_layout()
        plt.savefig(espectrograma_img_path)
        plt.close()  # Cerrar la figura para liberar memoria


# Ingresa la ruta de la carpeta que quieres explorar
ruta_carpeta = "C:/Users/Daniela Cuartas/OneDrive - Universidad de Antioquia/Tesis/Audios/Kiwi/Roroa Solos"

# Llama a la función para obtener la lista de archivos
lista_archivos = listar_archivos_en_carpeta(ruta_carpeta) #contiene todos los archivos .wav de todas las carpetas

# Separa los archivos según el género
archivos_male, archivos_female = separar_archivos_por_genero(lista_archivos)

# Ruta de la carpeta donde se guardarán los resultados
resultado_folder = "C:/Users/Daniela Cuartas/OneDrive - Universidad de Antioquia/Tesis/database_v3"

sumar_audios(archivos_male, archivos_female, resultado_folder)

# print(len(archivos_female)) 773
# print(len(archivos_male)) 2990