class Adapter(nn.Module):
    def __init__(
            self,
            input_dim,
            mid_dim
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.conv = nn.Conv3d(in_channels = mid_dim, out_channels = mid_dim, kernel_size=3, padding=1, groups=mid_dim)
        self.linear2 = nn.Linear(mid_dim, input_dim)

    def forward(self, features):
        out = self.linear1(features)
        out = F.relu(out)
        out = out.permute(0, 4, 1, 2, 3)
        out = self.conv(out)
        out = out.permute(0, 2, 3, 4, 1)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = features + out
        return out

