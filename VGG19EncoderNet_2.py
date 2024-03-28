# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:18:08 2024

@author: Daniela Cuartas
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary

# Architecture VGG19
VGG19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
# Then flatten and 4096x4096x1000 Linear Layers

# Class STFT and ISTFT was taken of Bermant, P.C. BioCPPNet: automatic bioacoustic source 
# separation with deep neural networks. Sci Rep 11, 23502 (2021). https://doi.org/10.1038/s41598-021-02790-2

# STFT class
class STFT(nn.Module): # ajustar la resolucion del espectrograma
    def __init__(self, kernel_size, stride, dB=False, epsilon=1e-8):
        super(STFT, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.window = nn.Parameter(torch.hann_window(kernel_size), requires_grad=False)
        self.epsilon = epsilon
        self.dB = dB

    def forward(self, x):
        S = torch.view_as_real(torch.stft(x.squeeze(dim=1), 
                                            n_fft=self.kernel_size, 
                                            hop_length=self.stride, 
                                            window=self.window, 
                                            return_complex=True))
        S_real = S[:, :, :, 0] + self.epsilon
        S_imag = S[:, :, :, 1] + self.epsilon
        P = torch.atan2(S_imag, S_real)
        D = torch.sqrt(torch.add(torch.pow(S_real, 2), torch.pow(S_imag, 2)))
        if self.dB:
            D = self.amplitude_to_db(D)
        return P, D

# ISTFT class
class iSTFT(nn.Module): 
    def __init__(self, kernel_size, stride, target):
        super(iSTFT, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.target = target  # Add objective length 
        self.window = nn.Parameter(torch.hann_window(kernel_size), requires_grad=False)

    def forward(self, S_complex):
        # S_complex is a complex tensor
        x = torch.istft(S_complex, n_fft=self.kernel_size, hop_length=self.stride, window=self.window, length=self.target).unsqueeze(dim=1)
        return x  

# Architecture class
class VGG19_net(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, n_fft=1024, hop_length=256, target_length=720000):  
        super(VGG19_net, self).__init__()
        self.input_channels = input_channels
        self.stft = STFT(kernel_size=n_fft, stride=hop_length)

#-------Architecture---------------
        self.encoder = self.create_encoder(VGG19) # Encoder
        self.decoder = self.create_decoder(VGG19) # Decoder        
        self.adjust_output = nn.Sequential(nn.ConvTranspose2d(64, 1, kernel_size=(3, 3), padding=(1, 1)),
        nn.Upsample(scale_factor=(1,2), mode='bilinear', align_corners=False))
        # self.final_conv = nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=(3, 3), padding=(1, 1))
        # self.upsample = nn.Upsample(size=(224, 448), mode='bilinear', align_corners=False)
        # self.conv = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(3, 3), padding=(1, 1))


        self.istft = iSTFT(kernel_size=n_fft, stride=hop_length, target=target_length)  

    def forward(self, x):
        # STFT
        P, D = self.stft(x) # P is phase, D is magnitude
        # Reshape D into 224x224, D.unsqueeze(1) add an extra dimension for being compatible with the network.
        # D=[batch_size, freq_bins, time_frames], then of unsqueeze D= [batch_size, 1, freq_bins, time_frames]

        # verificar el tamaño 513x32, debe ser 513 x miles(3000)

        D_unsqueezed = D.unsqueeze(1)  # [batch_size, 1, freq_bins, time_frames] tamaño torch.Size([1, 1, 513, 32])

        # Simulate 3-channel image by duplicating D_resized across the channel dimension
        D_3_channels = torch.cat([D_unsqueezed, D_unsqueezed, D_unsqueezed], dim=1) # [1, 3, 224, 224]

        # Cross D through the network 
        x = self.encoder(D_3_channels)
        x = self.decoder(x)
        x = adjust_output(x)

        # x = self.conv(x) # torch.Size([1, 1, 224, 448]) 1 channel with double of longitud
        
        # Divide tensor in 2 dimentions by width (ponerlo general en la mitad de la matriz)
        mask0 = x[:, :, :, :224]  # first half [1, 1, 224, 224]
        mask1 = x[:, :, :, 224:]  # second half [1, 1, 224, 224]

        mask0 = F.pad(mask0, (0,0,0,1)) # hace padding en esa dimension, temporal, pero daña la señal
        mask1 = F.pad(mask1, (0,0,0,1)) # hace padding en esa dimension
        # corregir para hacer la funcion de perdida para que evalue todo menos los ultimos donde esta malo debido al padding

        mask0 = mask0 * D # ([2, 1, 224, 224]) mascara es de 512 y D es de 513 problema, añadir padding a la mascara
        mask1 = mask1 * D

        # Reshape output to match with original size. Problema, el tamaño original es variable, con upsampling no deja usar squeeze
        # mag0 = self.increase_height(mask0)
        # mag0 = self.decrease_size(mag0)
        # mag0 = self.adjust(mag0)
        # mag1 = self.increase_height(mask1)
        # mag1 = self.decrease_size(mag1)
        # mag1 = self.adjust(mag1)

        #print(mag0.shape, P.shape)
        # Process audios separately
        audio1 = self.separate_audios(mask0, P) 
        audio2 = self.separate_audios(mask1, P)

        return audio1, audio2 

    def separate_audios(self, magnitude, P):
        # Create a complex tensor
        #print(magnitude.squeeze().shape, P.squeeze().shape)
        S_complex = torch.polar(magnitude.squeeze(), P.squeeze())
        
        # ISTFT to convert complex spectrum back to audio
        audio_reconstructed = self.istft(S_complex)

        return audio_reconstructed

    def create_encoder(self, architecture):
        layers = []
        in_channels = 3
        
        for x in architecture[:-1]:
            if type(x) == int:
                out_channels = x
                
                # Add convolutional layers
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                     nn.ReLU()]   
                in_channels = x

            # Add Maxpooling layers
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                
        return nn.Sequential(*layers)
    
    def create_decoder(self, architecture):
        layers = []
        prev_channels = 512
        
        for x in reversed(architecture[:-1]):
            if x == 'M':
                layers.append(nn.Upsample(scale_factor=2))
            elif type(x) == int:
                layers += [nn.ConvTranspose2d(in_channels=prev_channels, out_channels=x, 
                                              kernel_size=(3,3), padding=(1,1)),
                                              nn.ReLU()]
                prev_channels = x
    
        return nn.Sequential(*layers)
    

model = VGG19_net()

# # # Crear una instancia del modelo
# device = 'cuda' if torch.cuda.is_available() else 'cpu'        
# model = VGG19_net(input_channels=1, output_channels=1).to(device)

# # # # Imprimir la forma de la salida y un resumen del modelo
# x = torch.randn(1, 1, 8000).to(device)  # Ejemplo de señal de audio de 44100 muestras (1 segundo a 44100 Hz)
# audio1, audio2 = model(x)

# print(audio1.shape) #torch.Size([720000, 1])
#summary(model, (1, 8000))  # Señal de entrada con una sola muestra (1 canal) y 44100 muestras
            