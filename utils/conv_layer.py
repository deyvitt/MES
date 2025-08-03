# =============================================================================
# utils/conv_layer.py
# =============================================================================
import torch
import torch.nn as nn

class Mamba1DConv(nn.Module):
    def __init__(self, d_inner: int, d_conv: int = 4, bias: bool = True):
        super().__init__()
        self.d_conv = d_conv
        
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            bias=bias,
            groups=d_inner,  # Depthwise convolution
            padding=d_conv - 1
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_inner]
        Returns:
            x: [batch, seq_len, d_inner]
        """
        # Conv1d expects [batch, channels, seq_len]
        x = x.transpose(1, 2)  # [batch, d_inner, seq_len]
        x = self.conv1d(x)
        x = x[:, :, :-(self.d_conv-1)]  # Remove padding
        x = x.transpose(1, 2)  # [batch, seq_len, d_inner]
        return x 