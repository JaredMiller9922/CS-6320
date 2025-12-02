import torch
import torch.nn as nn

try:
   from torchvision.models import resnet18, ResNet18_Weights
   _NEW_API = True
except ImportError:
   from torchvision.models import resnet18
   _NEW_API = False


class MyResNet(nn.Module):
   def __init__(self):
      '''
      Init function to define the layers and loss function

      Note: Do not forget to freeze the layers of alexnet except the last one

      Download pretrained alexnet using pytorch's API (Hint: see the import
      statements)
      '''
      super().__init__()

      self.cnn_layers = nn.Sequential()
      self.fc_layers = nn.Sequential()
      self.loss_criterion = None

      ###########################################################################
      # Student code begin
      ###########################################################################
      # We will use softmax
      self.loss_criterion = nn.CrossEntropyLoss()

      # Get pretrained resnet
      self.model = resnet18(pretrained = True)

      for param in self.model.parameters():
         param.requires_grad = False

      # Update the last layer
      self.fc_layers = nn.Linear(512, 15)

      for param in self.fc_layers.parameters():
         param.requires_grad = True

      self.model.fc = self.fc_layers
      ###########################################################################
      # Student code end
      ###########################################################################

   def forward(self, x: torch.tensor) -> torch.tensor:
      '''
      Perform the forward pass with the net

      Args:
      -   x: the input image [Dim: (N,C,H,W)]
      Returns:
      -   y: the output (raw scores) of the net [Dim: (N,15)]
      '''

      model_output = None
      x = x.repeat(1, 3, 1, 1) # as AlexNet accepts color images

      ###########################################################################
      # Student code begin
      ###########################################################################
      model_output = self.model(x)
      ###########################################################################
      # Student code end
      ###########################################################################
      return model_output