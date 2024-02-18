import glob
import os
from pydub import AudioSegment
import random
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

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

def recortar_archivo(path, name):
    duracion_final = 90 * 1000  # Duración final en milisegundos (1 minuto y 30 segundos)
    audio = AudioSegment.from_file(path)

    # Seleccionar aleatoriamente entre 0 y 30 seg
    punto_inicio = random.choice([0, len(audio) - duracion_final])

    # Recortar el audio desde el punto de inicio hasta la duración final
    audio_recortado = audio[punto_inicio:punto_inicio + duracion_final] #0-1:30 / 30-2:00

    # Obtener el nombre del archivo original
    nombre_original = os.path.basename(path).split('.')[0]

    carpeta_destino = "C:/Users/Daniela Cuartas/OneDrive - Universidad de Antioquia/Tesis/recortes"

    resultado_name = f"{name}_{nombre_original}.wav"
    resultado_path = os.path.join(carpeta_destino, resultado_name)

    # Guardar el archivo de audio recortado
    audio_recortado.export(resultado_path, format="wav")

    return resultado_path

def sumar_audios(archivos_male, archivos_female):
    # Definir las carpetas base
    folder_base = "C:/Users/Daniela Cuartas/OneDrive - Universidad de Antioquia/Tesis/database_v3"
    folder_audio = os.path.join(folder_base, "audios")
    folder_espectrograma = os.path.join(folder_base, "espectrogramas")

    # Nueva subcarpeta
    subcarpeta = "male_female"

    # Actualizar las carpetas base con la nueva subcarpeta
    folder_audio_male_female = os.path.join(folder_audio, subcarpeta)
    folder_espectrograma_male_female = os.path.join(folder_espectrograma, subcarpeta)

    # Guarda male-female
    contadormf = 1
    for audio_male_path, audio_female_path in zip(archivos_male, archivos_female):
        # Recortar los archivos antes de cargarlos
        audio_male_recortado = recortar_archivo(audio_male_path, f"mf{contadormf:03d}") # devuelve la ruta del archivo recortado
        audio_female_recortado = recortar_archivo(audio_female_path, f"mf{contadormf:03d}")
        # Cargar ambos archivos de audio
        audio_male, sr1 = librosa.load(audio_male_recortado, sr=None)  
        audio_female, sr2 = librosa.load(audio_female_recortado, sr=None) 

        # Resamplear si es necesario
        if sr1 > sr2:
            audio_male = librosa.resample(y=audio_male, orig_sr=sr1, target_sr=sr2)
            sr1 = sr2
        else:
            audio_female = librosa.resample(y=audio_female, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1

        # Ajustar la longitud
        min_length = min(len(audio_male), len(audio_female))
        audio_male = audio_male[:min_length]
        audio_female = audio_female[:min_length]

        # Sumar los audios //constant energy pan
        audio_sumado = audio_male + audio_female

        # Extraer los nombres base sin la extensión y formar el nombre del archivo resultante
        base1 = os.path.basename(audio_male_path).split('.')[0]
        base2 = os.path.basename(audio_female_path).split('.')[0]
        
        resultado_name = f"mf{contadormf:03d}_{base1}_{base2}.wav"
        resultado_path = os.path.join(folder_audio_male_female, resultado_name)

        # Guardar el archivo de audio sumado
        sf.write(resultado_path, audio_sumado, sr1)

        # Calcular el espectrograma
        espectrograma = librosa.amplitude_to_db(np.abs(librosa.stft(audio_sumado)), ref=np.max)

        # Construir la ruta para guardar el espectrograma como imagen
        espectrograma_img_path = os.path.join(folder_espectrograma_male_female, f"mf{contadormf:03d}_{base1}_{base2}.png")
        contadormf+=1
        # Guardar el espectrograma como imagen
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(espectrograma, sr=sr1, x_axis='time', y_axis='linear')
        fig = plt.gcf()
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        #plt.tight_layout()
        plt.savefig(espectrograma_img_path, transparent=True)
        plt.close('all')  # Cerrar todas las figuras para liberar memoria

    # Nueva subcarpeta
    subcarpeta = "female_male"

    # Actualizar las carpetas base con la nueva subcarpeta
    folder_audio_female_male = os.path.join(folder_audio, subcarpeta)
    folder_espectrograma_female_male = os.path.join(folder_espectrograma, subcarpeta)
    
    # Guarda female-male
    contadorfm = 1
    for audio_female_path, audio_male_path in zip(archivos_female, archivos_male):
        # Recortar los archivos antes de cargarlos
        audio_female_recortado = recortar_archivo(audio_female_path, f"fm{contadorfm:03d}") # devuelve la ruta del archivo recortado
        audio_male_recortado = recortar_archivo(audio_male_path, f"fm{contadorfm:03d}")
        # Cargar ambos archivos de audio
        audio_female, sr1 = librosa.load(audio_female_recortado, sr=None)  
        audio_male, sr2 = librosa.load(audio_male_recortado, sr=None) 

        # Resamplear si es necesario
        if sr1 > sr2:
            audio_female = librosa.resample(y=audio_female, orig_sr=sr1, target_sr=sr2)
            sr1 = sr2
        else:
            audio_male = librosa.resample(y=audio_male, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1

        # Ajustar la longitud
        min_length = min(len(audio_male), len(audio_female))
        audio_female = audio_female[:min_length]
        audio_male = audio_male[:min_length]

        # Sumar los audios //constant energy pan
        audio_sumado = audio_female + audio_male

        # Extraer los nombres base sin la extensión y formar el nombre del archivo resultante
        base1 = os.path.basename(audio_female_path).split('.')[0]
        base2 = os.path.basename(audio_male_path).split('.')[0]
        resultado_name = f"fm{contadorfm:03d}_{base1}_{base2}.wav"
        resultado_path = os.path.join(folder_audio_female_male, resultado_name)

        # Guardar el archivo de audio sumado
        sf.write(resultado_path, audio_sumado, sr1)

        # Calcular el espectrograma
        espectrograma = librosa.amplitude_to_db(np.abs(librosa.stft(audio_sumado)), ref=np.max)

        # Construir la ruta para guardar el espectrograma como imagen
        espectrograma_img_path = os.path.join(folder_espectrograma_female_male, f"fm{contadorfm:03d}_{base1}_{base2}.png")
        contadorfm +=1

        # Guardar el espectrograma como imagen
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(espectrograma, sr=sr1, x_axis='time', y_axis='linear')
        fig = plt.gcf()
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        #plt.tight_layout()
        plt.savefig(espectrograma_img_path, transparent=True)
        plt.close('all')  # Cerrar todas las figuras para liberar memoria

    # Nueva subcarpeta
    subcarpeta = "male_male"

    # Actualizar las carpetas base con la nueva subcarpeta
    folder_audio_male_male = os.path.join(folder_audio, subcarpeta)
    folder_espectrograma_male_male = os.path.join(folder_espectrograma, subcarpeta)

    # Hacer una copia de la lista original para no modificarla directamente
    archivos_male_copia = archivos_male.copy()

    # Guardar male-male
    # Lista para almacenar las parejas
    parejas_male = []

    # Emparejar elementos hasta que ya no sea posible
    while len(archivos_male_copia) > 1:
        # Seleccionar aleatoriamente dos elementos sin repetición
        pareja = random.sample(archivos_male_copia, 2)

        # Agregar la pareja a la lista
        parejas_male.append(tuple(pareja))

        # Eliminar los elementos emparejados de la copia
        archivos_male_copia.remove(pareja[0])
        archivos_male_copia.remove(pareja[1])
    
    contadormm = 1
    for par in parejas_male:
        audio_male_path1, audio_male_path2 = par
        # Recortar los archivos antes de cargarlos
        audio_male_recortado1 = recortar_archivo(audio_male_path1, f"mm{contadormm:03d}") # devuelve la ruta del archivo recortado
        audio_male_recortado2 = recortar_archivo(audio_male_path2, f"mm{contadormm:03d}")
        # Cargar ambos archivos de audio
        audio_male1, sr1 = librosa.load(audio_male_recortado1, sr=None)  
        audio_male2, sr2 = librosa.load(audio_male_recortado2, sr=None) 

        # Resamplear si es necesario
        if sr1 > sr2:
            audio_male1 = librosa.resample(y=audio_male1, orig_sr=sr1, target_sr=sr2)
            sr1 = sr2
        else:
            audio_male2 = librosa.resample(y=audio_male2, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1

        # Ajustar la longitud
        min_length = min(len(audio_male1), len(audio_male2))
        audio_male1 = audio_male1[:min_length]
        audio_male2 = audio_male2[:min_length]

        # Sumar los audios //constant energy pan
        audio_sumado = audio_male1 + audio_male2

        # Extraer los nombres base sin la extensión y formar el nombre del archivo resultante
        base1 = os.path.basename(audio_male_path1).split('.')[0]
        base2 = os.path.basename(audio_male_path2).split('.')[0]
        resultado_name = f"mm{contadormm:03d}_{base1}_{base2}.wav"
        resultado_path = os.path.join(folder_audio_male_male, resultado_name)

        # Guardar el archivo de audio sumado
        sf.write(resultado_path, audio_sumado, sr1)

        # Calcular el espectrograma
        espectrograma = librosa.amplitude_to_db(np.abs(librosa.stft(audio_sumado)), ref=np.max)

        # Construir la ruta para guardar el espectrograma como imagen
        espectrograma_img_path = os.path.join(folder_espectrograma_male_male, f"mm{contadormm:03d}_{base1}_{base2}.png")
        contadormm += 1

        # Guardar el espectrograma como imagen
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(espectrograma, sr=sr1, x_axis='time', y_axis='linear')
        fig = plt.gcf()
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        #plt.tight_layout()
        plt.savefig(espectrograma_img_path, transparent=True)
        plt.close('all')  # Cerrar todas las figuras para liberar memoria

        # Nueva subcarpeta
    subcarpeta = "female_female"

    # Actualizar las carpetas base con la nueva subcarpeta
    folder_audio_female_female = os.path.join(folder_audio, subcarpeta)
    folder_espectrograma_female_female = os.path.join(folder_espectrograma, subcarpeta)

    # Hacer una copia de la lista original para no modificarla directamente
    archivos_female_copia = archivos_female.copy()

    # Guardar female-female
    # Lista para almacenar las parejas
    parejas_female = []

    # Emparejar elementos hasta que ya no sea posible
    while len(archivos_female_copia) > 1:
        # Seleccionar aleatoriamente dos elementos sin repetición
        pareja = random.sample(archivos_female_copia, 2)

        # Agregar la pareja a la lista
        parejas_female.append(tuple(pareja))

        # Eliminar los elementos emparejados de la copia
        archivos_female_copia.remove(pareja[0])
        archivos_female_copia.remove(pareja[1])
    
    contadorff = 1
    for par in parejas_female:
        audio_female_path1, audio_female_path2 = par
        # Recortar los archivos antes de cargarlos
        audio_female_recortado1 = recortar_archivo(audio_female_path1, f"ff{contadorff:03d}") # devuelve la ruta del archivo recortado
        audio_female_recortado2 = recortar_archivo(audio_female_path2, f"ff{contadorff:03d}")
        # Cargar ambos archivos de audio
        audio_female1, sr1 = librosa.load(audio_female_recortado1, sr=None)  
        audio_female2, sr2 = librosa.load(audio_female_recortado2, sr=None) 

        # Resamplear si es necesario
        if sr1 > sr2:
            audio_female1 = librosa.resample(y=audio_female1, orig_sr=sr1, target_sr=sr2)
            sr1 = sr2
        else:
            audio_female2 = librosa.resample(y=audio_female2, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1

        # Ajustar la longitud
        min_length = min(len(audio_female1), len(audio_female2))
        audio_female1 = audio_female1[:min_length]
        audio_female2 = audio_female2[:min_length]

        # Sumar los audios //constant energy pan
        audio_sumado = audio_female1 + audio_female2  #sqrt(alpha) *audio_female1 + sqrt(1-alpha)*audio_female2, alpha = 0.5

        # Extraer los nombres base sin la extensión y formar el nombre del archivo resultante
        base1 = os.path.basename(audio_female_path1).split('.')[0]
        base2 = os.path.basename(audio_female_path2).split('.')[0]
        resultado_name = f"ff{contadorff:03d}_{base1}_{base2}.wav"
        resultado_path = os.path.join(folder_audio_female_female, resultado_name)

        # Guardar el archivo de audio sumado
        sf.write(resultado_path, audio_sumado, sr1)

        # Calcular el espectrograma
        espectrograma = librosa.amplitude_to_db(np.abs(librosa.stft(audio_sumado)), ref=np.max)

        # Construir la ruta para guardar el espectrograma como imagen
        espectrograma_img_path = os.path.join(folder_espectrograma_female_female, f"ff{contadorff:03d}_{base1}_{base2}.png")
        contadorff += 1

        # Guardar el espectrograma como imagen
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(espectrograma, sr=sr1, x_axis='time', y_axis='linear')
        fig = plt.gcf()
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        #plt.tight_layout()
        plt.savefig(espectrograma_img_path, transparent=True)
        plt.close('all')  # Cerrar todas las figuras para liberar memoria


# ruta de la carpeta origen
ruta_carpeta = "C:/Users/Daniela Cuartas/OneDrive - Universidad de Antioquia/Tesis/Audios/Kiwi/Roroa Solos"

# Llama a la función para obtener la lista de archivos
lista_archivos = listar_archivos_en_carpeta(ruta_carpeta) #contiene todos los archivos .wav de todas las carpetas

# Separa los archivos según el género
archivos_male, archivos_female = separar_archivos_por_genero(lista_archivos)

sumar_audios(archivos_male, archivos_female)

# print(len(archivos_female)) 773
# print(len(archivos_male)) 2990