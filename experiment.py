import glob
import os
from pydub import AudioSegment
import random
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

def list_files_in_folder(path):
    # Use the pattern '**/*' to traverse all subfolders and files
    files = glob.glob(os.path.join(path, '**/*.wav'), recursive=True)
    # Filter only files, excluding directories
    files = [file for file in files if os.path.isfile(file)]
    
    return files

def separate_files_by_gender(wav_files):
    # Separate files with 'RoroaMale' in the filename
    male_files = [file for file in wav_files if "RoroaMale" in file]
    # Separate files with 'RoroaFemale' in the filename
    female_files = [file for file in wav_files if "RoroaFemale" in file]

    # Return male and female files
    return male_files, female_files

def trim_audio_file(path, name):
    final_duration = 90 * 1000  # Final duration in milliseconds (1 minute and 30 seconds)
    audio = AudioSegment.from_file(path)

    # Randomly select a starting point between 0 and 30 seconds
    start_point = random.choice([0, len(audio) - final_duration])

    # Trim the audio from the starting point to the final duration
    trimmed_audio = audio[start_point:start_point + final_duration]

    # Get the original file name
    original_name = os.path.basename(path).split('.')[0]

    destination_folder = "C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/recortes"

    result_name = f"{name}_{original_name}.wav"
    result_path = os.path.join(destination_folder, result_name)

    # Save the trimmed audio file
    trimmed_audio.export(result_path, format="wav")

    # Return path 
    return result_path

def trim_and_load_audio(audio_path, identifier):
    # Trim the audio file using the trim_audio_file function
    trimmed_path = trim_audio_file(audio_path, identifier)

    #  Load the trimmed audio file using librosa
    audio, sr = librosa.load(trimmed_path, sr=None)

    # Return the loaded audio data and its sampling rate
    return audio, sr

def resample_audio(audio1, audio2, sr1, sr2):
    # Check if the sampling rate of audio1 is greater than audio2
    if sr1 > sr2:
        # Resample audio1 to match the sampling rate of audio2
        audio1 = librosa.resample(y=audio1, orig_sr=sr1, target_sr=sr2)
        sr1 = sr2  # Update sr1 to match the resampled sampling rate
    else:
        # Resample audio2 to match the sampling rate of audio1
        audio2 = librosa.resample(y=audio2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1  # Update sr2 to match the resampled sampling rate
    
    # Return the resampled audio data and the updated sampling rates
    return audio1, audio2, sr1

def adjust_audio_length(audio1, audio2):
    # Determine the minimum length of the two audio signals
    min_length = min(len(audio1), len(audio2))

    # Return the adjusted audio signals
    return audio1[:min_length], audio2[:min_length]

def sum_audio_files(audio1, audio2):
    # Set the weight parameter (alpha) for the weighted sum
    alpha = 0.5
    
    # Calculate the square root of alpha and (1 - alpha)
    sqrt_alpha = np.sqrt(alpha)
    sqrt_one_minus_alpha = np.sqrt(1 - alpha)
    
    # Perform the weighted sum of the two audio signals
    result_audio = sqrt_alpha * audio1 + sqrt_one_minus_alpha * audio2
    
    # Return the result of the weighted sum
    return result_audio

def process_audio_pair(arch1, arch2, folder_audio, folder_spectrogram, prefix, counter):
    # Trim and load audio
    audio1, sr1 = trim_and_load_audio(arch1, f"{prefix}{counter:03d}")
    audio2, sr2 = trim_and_load_audio(arch2, f"{prefix}{counter:03d}")

    # Resample audio to have the same sampling rate and adjust lengths
    audio1, audio2, sr1 = resample_audio(audio1, audio2, sr1, sr2)
    audio1, audio2 = adjust_audio_length(audio1, audio2)

    # Sum audios
    audio_sum = sum_audio_files(audio1, audio2)

    # Generate a unique result name based on the input filenames and counter
    base1 = os.path.basename(arch1).split('.')[0]
    base2 = os.path.basename(arch2).split('.')[0]
    result_name = f"{prefix}{counter:03d}_{base1}_{base2}.wav"
    result_path = os.path.join(folder_audio, result_name)

    # Write the summed audio to a new WAV file
    sf.write(result_path, audio_sum, sr1)
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio_sum)), ref=np.max)
    spectrogram_img_path = os.path.join(folder_spectrogram, f"{prefix}{counter:03d}_{base1}_{base2}.png")
    
    # Plot and save the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr1, x_axis='time', y_axis='linear')
    fig = plt.gcf()
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    plt.savefig(spectrogram_img_path, transparent=True)
    plt.close('all')  # Close figures to prevent memory leaks

def same_pairs():
    # Initialize an empty list to store pairs
    pairs = []

    # Continue until there is only one element left in the list
    while len(copy_files1) > 1:
        # Randomly select two elements without replacement from copy_files1
        pair = random.sample(copy_files1, 2)

        # Append the selected pair as a tuple to the pairs list
        pairs.append(tuple(pair))

        # Remove the selected elements from copy_files1
        copy_files1.remove(pair[0])
        copy_files1.remove(pair[1])

    # Return the list of pairs
    return pairs


def process_and_sum_pairs(files_1, files_2, folder_audio, folder_spectrogram, prefix, ban):
    # Initialize a counter for naming processed pairs
    counter = 1

    if ban == 0:
        # Process pairs male-female, female-male
        for file1, file2 in zip(files_1, files_2):
            process_audio_pair(file1, file2, folder_audio, folder_spectrogram, prefix, counter)
            counter += 1
    else:
        # If ban is not equal to 0, create a copy of files_1
        global copy_files1
        copy_files1 = files_1.copy()

        # Generate pairs from the copied list
        pairs = same_pairs()

        # Process pairs male-male, female-female
        for pair in pairs:
            file1, file2 = pair
            process_audio_pair(file1, file2, folder_audio, folder_spectrogram, prefix, counter)  
            counter += 1   
    
def sum_audio_files_and_generate_spectrograms_for_each_category(files_male, files_female):
    # Define the base folder for the database
    folder_base = "C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3"

    # Define subfolders for audio and spectrogram storage
    folder_audio = os.path.join(folder_base, "audios")
    folder_spectrogram = os.path.join(folder_base, "espectrogramas")

    # Define subfolder names for different categories
    subfolder_male_female = "male_female"
    subfolder_female_male = "female_male"
    subfolder_male_male = "male_male"
    subfolder_female_female = "female_female"
 
    # Define paths for male-female category
    folder_audio_male_female = os.path.join(folder_audio, subfolder_male_female)
    folder_spectrogram_male_female = os.path.join(folder_spectrogram, subfolder_male_female)

    # Process and sum pairs for male-female category
    process_and_sum_pairs(files_male, files_female, folder_audio_male_female, folder_spectrogram_male_female, "mf", 0)

    # Define paths for female-male category
    folder_audio_female_male = os.path.join(folder_audio, subfolder_female_male)
    folder_spectrogram_female_male = os.path.join(folder_spectrogram, subfolder_female_male)
    
    # Process and sum pairs for female-male category
    process_and_sum_pairs(files_female, files_male, folder_audio_female_male, folder_spectrogram_female_male, "fm", 0)
    
    # Define paths for male-male category
    folder_audio_male_male = os.path.join(folder_audio, subfolder_male_male)
    folder_spectrogram_male_male = os.path.join(folder_spectrogram, subfolder_male_male)

    # Process and sum pairs for male-male category
    process_and_sum_pairs(files_male, files_male, folder_audio_male_male, folder_spectrogram_male_male, "mm", 1)
    
    # Define paths for female-female category
    folder_audio_female_female = os.path.join(folder_audio, subfolder_female_female)
    folder_spectrogram_female_female = os.path.join(folder_spectrogram, subfolder_female_female)

    # Process and sum pairs for female-female category
    process_and_sum_pairs(files_female, files_female, folder_audio_female_female, folder_spectrogram_female_female, "ff", 2)


origin_path = "C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/Audios/Kiwi/Roroa Solos"

# Get a list of all .wav files in the specified folder
file_list = list_files_in_folder(origin_path) 

# Separate the files into two lists based on gender
files_male, files_female = separate_files_by_gender(file_list)

# Process and sum audio files, and generate spectrograms for each category
sum_audio_files_and_generate_spectrograms_for_each_category(files_male, files_female)

# print(len(files_female)) 773
# print(len(files_male)) 2990