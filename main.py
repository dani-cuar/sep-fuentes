from Train import *
from Validate import *
from VGG19EncoderNet_1 import *
from GenerateDataloader import *
from torch.utils.data import DataLoader, random_split

mixed_folder = "C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/audios" 
y1_folder = "C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/audios separados/y1" 
y2_folder = "C:/Users/Daniela Cuartas/Documents/UdeA/Proyecto de grado/database_v3/audios separados/y2"

x_files, y1_files, y2_files = list_mixed_and_separated_files(mixed_folder, y1_folder, y2_folder)
complete_dataset = AudioDataset(x_files, y1_files, y2_files)

total_size = len(complete_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, validation_dataset = random_split(complete_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)  # batch_size=1 para la generación de espectrogramas

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device selected: {device}")

if device.type == 'cuda':
    torch.cuda.empty_cache()

model = VGG19_net()
# train
train(train_loader, model, device, 5, 16, 0.0001, 0.9)

# Validation and generation of spectrograms after to complete all apochs 
model.load_state_dict(torch.load('modelos/VGG19_model.pth'))
validate_and_save_spectrograms(model, validation_loader, device)