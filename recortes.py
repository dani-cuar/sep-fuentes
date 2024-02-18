import os
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Configuración de directorios
dir_txt = 'C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/Kokako_selections_Duets_Only'
dir_wav = 'C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/Kokako_singer_separation'
output_base_dir = 'C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/Duets_Clips'
spectrogram_dir = 'C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/Duets_Spectrograms'

# Asegurarse de que los directorios de salida existen
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)
if not os.path.exists(spectrogram_dir):
    os.makedirs(spectrogram_dir)

# Función para encontrar el archivo de audio correspondiente
def find_audio_file(txt_filename):
    audio_file_prefix = txt_filename.split('_')[0]  # Asume que el prefijo del archivo de audio es el primer elemento del nombre
    for root, dirs, files in os.walk(dir_wav):
        for file in files:
            if file.startswith(audio_file_prefix) and file.endswith('.wav'):
                return os.path.join(root, file)
    return None

# Función para generar y guardar el espectrograma de un clip de audio
def save_spectrogram(audio_clip, filename):
    # Convertir AudioSegment a un arreglo de numpy
    samples = np.array(audio_clip.get_array_of_samples())
    fs = audio_clip.frame_rate
    
    # Generar el espectrograma
    f, t, Sxx = spectrogram(samples, fs)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    
    # Guardar la figura
    plt.savefig(filename)
    plt.close()  # Cerrar la figura para liberar memoria

# Función para procesar los archivos .txt y recortar los audios
def process_txt_files():
    for txt_file in os.listdir(dir_txt):
        if txt_file.endswith('.txt'):
            audio_file_path = find_audio_file(txt_file)
            if audio_file_path:
                audio = AudioSegment.from_wav(audio_file_path)
                output_dir = os.path.join(output_base_dir, txt_file.replace('.selections.txt', ''))
                spectrogram_output_dir = os.path.join(spectrogram_dir, txt_file.replace('.selections.txt', ''))
                
                # Crear directorios si no existen
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if not os.path.exists(spectrogram_output_dir):
                    os.makedirs(spectrogram_output_dir)
                
                with open(os.path.join(dir_txt, txt_file), 'r') as f:
                    for line in f.readlines():
                        parts = line.split('\t')
                        if 'Duet' in parts:
                            start_time = float(parts[3]) * 1000  # Convertir a milisegundos
                            end_time = float(parts[4]) * 1000    # Convertir a milisegundos
                            clip = audio[start_time:end_time]
                            # Convertir a mono
                            clip = clip.set_channels(1)
                            clip_name = f"{txt_file.split('.')[0]}_{int(start_time)}_{int(end_time)}"
                            clip_path = os.path.join(output_dir, clip_name + ".wav")
                            spectrogram_path = os.path.join(spectrogram_output_dir, clip_name + ".png")
                            
                            
                            # Exportar clip de audio
                            clip.export(clip_path, format="wav", parameters=["-acodec", "pcm_s32le"])
                            
                            # Generar y guardar espectrograma
                            save_spectrogram(clip, spectrogram_path)
                            #print(f"Clip y espectrograma guardados: {clip_path}, {spectrogram_path}")

process_txt_files()
