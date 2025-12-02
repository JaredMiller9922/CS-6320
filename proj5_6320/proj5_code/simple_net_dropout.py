import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention
    to understand what it means
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

    # Default Values
    self.cnn_layers = nn.Sequential(
      nn.Conv2d(1,10,5),
      # BatchNorm takes the out_ch
      nn.BatchNorm2d(10),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(10,20,5),
      # BatchNorm takes the out_ch
      nn.BatchNorm2d(20),
      nn.ReLU(),
      nn.MaxPool2d(3),
    )

    # Experimental Values
    # self.cnn_layers = nn.Sequential(
      # nn.Conv2d(1,15,3),
      # nn.MaxPool2d(3),
      # nn.ReLU(),
      # nn.Conv2d(10,25,3),
      # nn.MaxPool2d(3),
      # nn.ReLU(),
    # )

    # We can actually compute our hidden layer size dynamically
    with torch.no_grad():
      dummy = torch.zeros(1, 1, 64, 64)
      conv_out = self.cnn_layers(dummy)
      flat_dim = conv_out.numel()

    # Default Values
    self.fc_layers = nn.Sequential(
      nn.Linear(flat_dim,150),
      nn.ReLU(),
      nn.Dropout(),
      nn.Linear(150, 15),
    )

    # Experiemental Values
    # self.fc_layers = nn.Sequential(
      # nn.Linear(flat_dim,10),
      # nn.ReLU(),
      # nn.Linear(10, 15),
    # )


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
    ###########################################################################
    # Student code begin
    ###########################################################################

    x = self.cnn_layers(x)
    # x.view acts the same as x.reshape
    x = x.view(x.size(0), -1)

    x = self.fc_layers(x)
    model_output = x

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output
