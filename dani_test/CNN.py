import torch
import torch.nn as nn  # nn for neural network

# Couche de convolution réutilisable
# Utilise un filtre (kernel de 3x3)
# Stride : nombre de pixel que le filtre saute à chaque déplacement. Nous utilisons un stride de 2 pour faire une réduction de la taille
# bn : nous allons normaliser les valeurs pour rendre l'entraînement plus stable
# dropout : probabilité de mettre des neurones égales à 0, pour éviter le sur-apprentissage
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, batch_norm = True, dropout = 0.3):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(), # normaliser si activer, sinon ne rien faire
            nn.ReLU(),  # couche d'activation pour non-linéarité
            nn.Dropout(dropout)
        )
        
    def forward(self,x):
        return self.block(x)
        
class CNN(nn.Module):
    # Specify the number of classes for this task
    def __init__(self, classes, dropout = 0.4, strides = None, kernel_size = 3):
        super().__init__()

        # Couche convolutionnelle
        # Ajouter trois couches convolutionnelles pour retrouver le plus de caractéristiques
        self.conv_layers = nn.Sequential(
            ConvBlock(4,32, stride = 1),
            nn.MaxPool2d(2),
            ConvBlock(32,64, stride = 1),
            nn.MaxPool2d(2),
            ConvBlock(64,128, stride = 1),
            nn.MaxPool2d(2),
            ConvBlock(128,256, stride = 1)   # le modèle apprend 256 caractérisitiques
            ConvBlock(256,512)
        )

        # Réduction spatiale
        self.aggregation = nn.AdaptiveAvgPool2d(1)

        # Partie fully-connected
        self.mlp = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Dropour(0.2),
            nn.Linear(128,classes)

            #nn.Linear(128,64),
            #nn.ReLU(),
            #nn.Dropout(dropout),

            #nn.Linear(64, classes)
        )

    def forward(self, x):
        y = self.conv_layers(x)  
        y = self.aggregation(y)
        y = torch.flatten(y,1)
        y = self.mlp(y)
        return y