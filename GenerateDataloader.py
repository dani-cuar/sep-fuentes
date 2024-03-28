
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch
import torchaudio

def list_mixed_and_separated_files(mixed_folder, y1_folder, y2_folder):
    x_files = []
    y1_files = []
    y2_files = []

    # Recursively cycle through the mixed audio folder
    for root, dirs, files in os.walk(mixed_folder):
        for file in files:
            if file.endswith('.wav'):
                mixed_path = os.path.join(root, file)
                x_files.append(mixed_path)

    # cycle through the y1 audio folder
    for file in os.listdir(y1_folder):
        if file.endswith('.wav'):
            y1_path = os.path.join(y1_folder, file)
            y1_files.append(y1_path)
    
    # cycle through the y2 audio folder
    for file in os.listdir(y2_folder):
        if file.endswith('.wav'):
            y2_path = os.path.join(y2_folder, file)
            y2_files.append(y2_path)
                
    return x_files, y1_files, y2_files

class AudioDataset(Dataset):
    def __init__(self, x_files, y1_files, y2_files):
        self.x_files = x_files
        self.y1_files = y1_files
        self.y2_files = y2_files

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        target_sample_rate = 8000

        # Load and possibly resample the mixed audio file
        x_file = self.x_files[idx]
        x_waveform, sample_rate = torchaudio.load(x_file) # Generate tensors
        if sample_rate != target_sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            x_waveform = resample_transform(x_waveform)

        # Load and possibly resample the mixed audio file
        y1_file = self.y1_files[idx]
        y1_waveform, sample_rate = torchaudio.load(y1_file) # Generate tensors
        if sample_rate != target_sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            y1_waveform = resample_transform(y1_waveform)

        # Load and possibly resample the mixed audio file
        y2_file = self.y2_files[idx]
        y2_waveform, sample_rate = torchaudio.load(y2_file) # Generate tensors
        if sample_rate != target_sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            y2_waveform = resample_transform(y2_waveform)        

        return x_waveform, y1_waveform, y2_waveform


# Uso de las funciones
# Ejemplo de uso:
# mixed_folder = 'C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/audios'
# y1_folder = 'C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/audios separados/y1'
# y2_folder = 'C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/audios separados/y2'
# x_files, y1_files, y2_files = list_mixed_and_separated_files(mixed_folder, y1_folder, y2_folder)

# # Creación del conjunto de datos
# complete_dataset = AudioDataset(x_files, y1_files, y2_files)
# # # Split the dataset
# total_size = len(complete_dataset)
# train_size = int(0.8 * total_size)
# val_size = total_size - train_size

# train_dataset, validation_dataset = random_split(complete_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# for batch in train_loader:
#     # `batch` contiene tus datos y etiquetas para este lote
#     # Descomponemos el lote en datos (waveforms) y etiquetas
#     waveforms, output1, output2 = batch
    
# #     # Ahora puedes hacer lo que necesites con estos datos
# #     # Por ejemplo, imprimir la forma del primer waveform y su etiqueta correspondiente
#     print("Waveform shape:", waveforms[1].shape)
#     print("output1:", output1[0])
#     print("output2:", output2[0])
# #     # Rompemos después de la primera iteración para solo ver el primer lote
#     break