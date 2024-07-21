class Local_cnn(nn.Module):

    def __init__(self, in_features, kernel_size = 3, k = 1.5, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        '''n_embd,  # dimension of the input features
        kernel_size = 3,  # conv kernel size
        n_ds_stride = 1,  # downsampling stride for the current layer
        k = 1.5,  # k
        group = 1,  # group for cnn
        n_out = None,  # output dimension, if None, set to input dim
        n_hidden = None,  # hidden dim for mlp
        path_pdrop = 0.0,  # drop path rate
        act_layer = nn.GELU,  # nonlinear activation used after conv, default ReLU,
        downsample_type = 'max',
        init_conv_vars = 1  # init gaussian variance for the weight
        Local_cnn(in_features=dim, hidden_features=mlp_hidden_dim,
                                                             act_layer=act_layer, drop=drop)'''

        super().__init__()
        '''out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.TC = nn.Conv1d(hidden_features, hidden_features, 3, 1, 1, bias=True,
                            groups=hidden_features)  # k=3, stride=1, padding=1
        self.act = act_layer()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)'''

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(in_features, in_features, kernel_size, stride=1, padding=kernel_size // 2, groups=in_features)
        self.fc = nn.Conv1d(in_features, in_features, 1, stride=1, padding=0, groups=in_features)
        self.convw = nn.Conv1d(in_features, in_features, kernel_size, stride=1, padding=kernel_size // 2, groups=in_features)
        self.convkw = nn.Conv1d(in_features, in_features, up_size, stride=1, padding=up_size // 2, groups=in_features)
        self.global_fc = nn.Conv1d(in_features, in_features, 1, stride=1, padding=0, groups=in_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        '''x = self.linear1(x)
        x = x.transpose(1, 2)
        x = self.TC(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)'''
        out = x.transpose(1,2)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.convkw(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        x = fc * phi + (convw + convkw) * psi + out
        x = x.transpose(1,2)
        return x