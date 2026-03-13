
import torch
import torch.nn as nn


class Resnet_projector(nn.Module):

    def __init__(self, base_encoder, dim=128):
        """
        dim: feature dimension (default: 128)
        """
        super(Resnet_projector, self).__init__()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_k = base_encoder(num_classes=dim)

        dim_mlp = self.encoder_k.fc.weight.shape[1]
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
        )


    def forward(self, im_q):
        """
        Input:
            im_q: a batch of query images
        Output:
            feature
        """

        return self.encoder_k(im_q)
