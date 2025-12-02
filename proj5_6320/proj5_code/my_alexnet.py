import torch
import torch.nn as nn

try:
    from torchvision.models import alexnet, AlexNet_Weights
    _NEW_API = True
except ImportError:
    from torchvision.models import alexnet
    _NEW_API = False


class MyAlexNet(nn.Module):
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

    # Get pretrained AlexNet
    self.model = alexnet(pretrained = True)

    # Match convolutional layers of CNN
    self.cnn_layers = self.model.features

    #.classifier.children returns each layer of the network
    default_classifier = list(self.model.classifier.children())

    # Update the last layer
    new_classifier = default_classifier[:-1]
    new_classifier.append(nn.Linear(4096,15))

    # Convert back to sequential
    self.fc_layers = nn.Sequential(*new_classifier)

    # Start by freezing all parameters
    for param in self.fc_layers.parameters():
       param.requires_grad = False

    # Start by freezing all parameters
    for param in self.cnn_layers.parameters():
       param.requires_grad = False
      
    # Unfreeze the final layer
    for param in self.fc_layers[-1].parameters():
       param.requires_grad = True

    self.model.classifier = self.fc_layers

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
    # Run the forward pass
    x = self.cnn_layers(x)
    # x.view acts the same as x.reshape
    x = x.view(x.size(0), -1)

    x = self.fc_layers(x)
    model_output = x
    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output