import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import numpy as np
from mir_eval.separation import bss_eval_sources

def generate_and_save_spectrogram(audio, fs, file_name):
    # STFT
    spectrogram = torch.stft(audio, n_fft=2048, hop_length=512, window=torch.hann_window(2048),
                             return_complex=True)
    
    # Calculate magnitude of spectrogram
    spectrogram_magnitude = torch.abs(spectrogram).numpy()
    
    # Convert to decibels
    spectrogram_db = 20 * np.log10(spectrogram_magnitude + 1e-6)  
    
    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_db[0], aspect='auto', origin='lower', cmap='viridis', extent=[0, spectrogram_db.shape[-1], 0, fs / 2])
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time Frame')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def validate_and_save_spectrograms(model, validation_loader, device, fs=8000):
    spectrogram_folder = './spectrograms'
    if not os.path.exists(spectrogram_folder):
        os.makedirs(spectrogram_folder)
    
    model.to(device)
    model.eval()
    total_loss = 0
    total_sdr = 0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            inputs, output1, output2 = [d.to(device) for d in data]
            prediction1, prediction2 = model(inputs)
            
            # Make sure predictions and goals have the same form
            prediction1 = prediction1.view_as(output1)
            prediction2 = prediction2.view_as(output2)

            # Convert tensors to numpy for evaluation with mir_eval
            ref1 = output1.squeeze(0).cpu().numpy()  
            est1 = prediction1.squeeze(0).cpu().numpy()
            ref2 = output2.squeeze(0).cpu().numpy()
            est2 = prediction2.squeeze(0).cpu().numpy()

            # Calculate SDR for each pair
            sdr1, _, _, _ = bss_eval_sources(ref1, est1)
            sdr2, _, _, _ = bss_eval_sources(ref2, est2)
            total_sdr += (np.mean(sdr1) + np.mean(sdr2)) / 2

            loss1 = loss_fn(prediction1, output1)
            loss2 = loss_fn(prediction2, output2)
            loss = (loss1 + loss2) / 2
            total_loss += loss.item()
            
            if i == len(validation_loader) - 2:
                os.makedirs(spectrogram_folder, exist_ok=True)
                file_name1 = os.path.join(spectrogram_folder, 'prediction1.png')
                generate_and_save_spectrogram(prediction1[0].cpu(), fs, file_name1)
                
                file_name2 = os.path.join(spectrogram_folder, 'output1.png')
                generate_and_save_spectrogram(output1[0].cpu(), fs, file_name2)

                file_name3 = os.path.join(spectrogram_folder, 'prediction2.png')
                generate_and_save_spectrogram(prediction2[0].cpu(), fs, file_name3)

                file_name4 = os.path.join(spectrogram_folder, 'output2.png')
                generate_and_save_spectrogram(output2[0].cpu(), fs, file_name4)

                file_name5 = os.path.join(spectrogram_folder, 'input.png')
                generate_and_save_spectrogram(inputs[0].cpu(), fs, file_name5)

    average_loss = total_loss / len(validation_loader)
    print(f'Validation completed. Average loss: {average_loss:.4f}')
    avg_sdr = total_sdr / len(validation_loader)
    print(f'Average SDR: {avg_sdr:.4f}')