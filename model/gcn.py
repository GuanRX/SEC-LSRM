from torch import nn
from torch_geometric.nn import GCNConv


class GraphAutoencoderBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden = GCNConv(in_channels, hidden_channels)
        self.output = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        hidden = self.hidden(x, edge_index, edge_weight).relu()
        return self.output(hidden, edge_index, edge_weight)
