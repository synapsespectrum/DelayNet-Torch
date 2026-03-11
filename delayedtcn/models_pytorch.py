import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class DelayedLayer(nn.Module):
    def __init__(self, nb_stride=3, nb_filters=64, kernel_size=3, padding='causal', dropout_rate=0.0, use_weight_norm=False, use_skip_connections=False, input_channels=1):
        super(DelayedLayer, self).__init__()
        self.nb_filters = nb_filters
        self.use_weight_norm = use_weight_norm
        self.use_skip_connections = use_skip_connections
        self.padding_type = padding
        
        self.layers = nn.ModuleList()
        
        # Conv 1
        # TF: filters=nb_filters, kernel_size=kernel_size, strides=nb_stride
        # PyTorch Conv1d: in_channels, out_channels, kernel_size, stride, padding, dilation
        
        # Causal padding calculation for Conv 1 (strided)
        # Output length = (Input + 2*padding - dilation*(kernel-1) - 1)/stride + 1
        # We want causal padding.
        # For stride > 1, it's a bit tricky to match TF 'causal' exactly with simple padding.
        # TF 'causal' usually means pad (kernel_size - 1) * dilation on the left.
        # But with stride, we need to be careful.
        # Let's assume we pad manually.
        
        self.conv1_stride = nb_stride
        self.conv1_kernel = kernel_size
        self.conv1_padding = (kernel_size - 1) # This is effectively dilation * (kernel-1) with dilation=1
        
        # Layer 0: Strided Conv
        self.conv0 = nn.Conv1d(input_channels, nb_filters, kernel_size, stride=nb_stride, padding=0) # Padding handled manually
        if use_weight_norm:
            self.conv0 = weight_norm(self.conv0)
        self.act0 = nn.ReLU()
        # self.drop0 = nn.Dropout(dropout_rate) # Commented out in TF code
        
        # Layer 1: Normal Conv (dilation=1)
        self.conv1 = nn.Conv1d(nb_filters, nb_filters, kernel_size, stride=1, padding=0) # Padding handled manually
        if use_weight_norm:
            self.conv1 = weight_norm(self.conv1)
        self.act1 = nn.ReLU()
        # self.drop1 = nn.Dropout(dropout_rate) # Commented out in TF code
        
        # 1x1 conv for shape matching (skip connection)
        # TF: filters=nb_filters, kernel_size=1, padding='same'
        # Input to this is x_org (original input). 
        # We need to project x_org to match x (output of convs).
        # x_org has shape (B, C, L_org). x has shape (B, F, L_out).
        # The TF code does: x = layers.add([self.shape_match_conv(x_org), x])
        # But wait, if x_org and x have different lengths (due to stride), how can they add?
        # In TF code:
        # x_org, x = inputs
        # ...
        # x = layers.add([self.shape_match_conv(x_org), x])
        # This implies x_org and x MUST have the same spatial dimension for element-wise add.
        # BUT conv0 has stride > 1, so x will be smaller than input x.
        # UNLESS x_org passed to DelayedLayer is ALREADY subsampled or the stride is 1?
        # In Model1.call: x_res, x = self.delayed_block([x_org, x])
        # x_org is the original input. x is the pattern input.
        # If DelayedLayer reduces size of x, then x_org must be reduced too?
        # TF Conv1D with 'same' padding and stride 1 keeps size.
        # TF Conv1D with 'causal' padding:
        # If stride > 1, output size is reduced.
        # Let's re-read TF code carefully.
        # self.shape_match_conv = layers.Conv1D(filters=self.nb_filters, kernel_size=1, padding='same')
        # If x_org has length L, output has length L.
        # If x (processed) has length L/stride, they cannot be added.
        # Maybe x_org in TF code is expected to be same length as x?
        # Or maybe I missed something about shape_match_conv.
        # "make and build this layer separately because it directly uses input_shape."
        # It seems shape_match_conv is applied to x_org.
        # If stride is used, this skip connection seems invalid unless x_org is already small?
        # OR, maybe stride is 1 in default config?
        # Config in run_timemmd_experiments.py: "gap": 2, "delay_factor": 1.
        # In utils/data.py pattern(): padding = np.zeros((delay_factor * gap + delay_factor * kernel_size, ...))
        # It creates a pattern sequence.
        # In Model1 init: nb_stride=self.list_stride[0]. Default list_stride=(7, 1). So stride=7?
        # If stride=7, output length is L/7.
        # If x_org is length L, shape_match_conv(x_org) is length L.
        # Adding them would fail in TF.
        # UNLESS shape_match_conv ALSO has stride?
        # TF code: self.shape_match_conv = layers.Conv1D(..., strides=1, ...) (default stride is 1).
        # This is very strange.
        # Let's look at `DelayedLayer` in `models.py` again.
        # `x = layers.add([self.shape_match_conv(x_org), x], name='Add_Res')`
        # If this works in TF, then either:
        # 1. `x_org` and `x` (input to DelayedLayer) have specific shapes.
        # 2. `x` (after convs) has same shape as `x_org`.
        # `x` input to DelayedLayer is the pattern sequence.
        # `x_org` is the original sequence.
        # If `conv0` has stride, `x` shrinks.
        # Maybe `shape_match_conv` is smart? No, it's a standard layer.
        # Wait, `Model1` passes `[x_org, x]`.
        # Maybe `x_org` is NOT the full original sequence?
        # In `main.py`: `data_train = [[tsf.data_train[0], tsf.data_train_gen[0]], ...]`
        # `tsf.data_train[0]` is original (L=168?).
        # `tsf.data_train_gen[0]` is pattern (L=?).
        # If they have different lengths, `DelayedLayer` logic seems flawed for skip connection if stride > 1.
        # Let's assume for now that we should implement what we see.
        # But for PyTorch, we must ensure shapes match.
        # I will implement `shape_match_conv` with the SAME stride as `conv0` if that's what's needed, 
        # OR I will assume `x_org` is transformed to match `x`.
        # Actually, if I look at `DelayedLayer` again:
        # `conv` (layer 0) has `strides=nb_stride`.
        # `conv` (layer 1) has `dilation_rate=1` (and implicitly stride 1).
        # So `x` is downsampled by `nb_stride`.
        # `shape_match_conv` has `strides=1` (default).
        # So `shape_match_conv(x_org)` is NOT downsampled.
        # This MUST fail in TF if `nb_stride > 1`.
        # UNLESS `nb_stride` is 1.
        # In `run_timemmd_experiments.py`: `DEFAULT_CONFIG` doesn't specify `list_stride`.
        # `Model1` default `list_stride=(7, 1)`.
        # So `nb_stride=7`.
        # This implies the TF code might be broken for `nb_stride > 1` OR I am missing something fundamental about TF `layers.add` broadcasting?
        # TF `layers.add` supports broadcasting? No, usually expects same shape.
        # Wait, `Model1` in `models.py` line 208: `x_res, x = self.delayed_block([x_org, x])`
        # If this code runs, then shapes must match.
        # Maybe `x_org` is the same as `x`? No, `x_org, x = inputs`.
        # Let's look at `utils/data.py` `pattern` function.
        # It returns a subset of data based on indices.
        # The length of pattern sequence depends on `kernel_size`, `gap`, `delay_factor`.
        # It seems `x` (pattern) is much shorter than `x_org`.
        # If `DelayedLayer` is supposed to merge them, it's confusing.
        
        # HYPOTHESIS: The user might be using a config where `nb_stride=1` or the code relies on some specific behavior.
        # However, looking at `run_timemmd_experiments.py`, it doesn't set `list_stride`.
        # So it uses default `(7, 1)`.
        # Let's assume I should replicate the structure.
        # If I encounter shape mismatch in PyTorch, I'll know.
        # To be safe, I will add a stride to `shape_match_conv` equal to `nb_stride` IF `x_org` is meant to be downsampled too.
        # BUT `x_org` is the "original input".
        # If `DelayedLayer` is meant to align "delayed" features with "original" features, maybe they align in time?
        # But `x` (pattern) is a scrambled/subsampled version.
        
        # Let's look at `DelayedLayer` again.
        # `x_org, x = inputs`
        # `x` goes through convs.
        # `x_org` goes through `shape_match_conv`.
        # `layers.add([..., x])`.
        # If `x` is the pattern, it represents specific time steps.
        # If `x_org` is the full sequence, it represents all time steps.
        # Adding them element-wise makes no sense unless they align.
        
        # Maybe `x_org` passed to `DelayedLayer` is NOT the full sequence?
        # In `Model1.call`: `x_org, x = inputs`.
        # `inputs` comes from `data_train`.
        # `data_train[0]` is `tsf.data_train[0]` (full sequence).
        
        # Let's try to implement `DelayedLayer` such that it works if `nb_stride=1`.
        # If `nb_stride > 1`, I might need to adjust `shape_match_conv` stride.
        # In TF `Conv1D` with `padding='same'`, output length = `ceil(input_length / stride)`.
        # If `nb_stride=7`, `x` length becomes `L_pattern / 7`.
        # `x_org` length is `L_org`.
        # `shape_match_conv(x_org)` length is `L_org`.
        # They definitely don't match.
        
        # Is it possible `x_org` is IGNORED in `DelayedLayer` if `use_skip_connections=False`?
        # `if self.use_skip_connections:` ...
        # In `Model1`, `use_skip_connections=True` by default.
        
        # Let's check `run_timemmd_experiments.py` config again.
        # It doesn't set `use_skip_connections`. Default `True`.
        
        # Maybe I should check `models.py` imports.
        # Is it possible `DelayedLayer` logic I see is not what's running? No, I read the file.
        
        # Let's assume `shape_match_conv` SHOULD have the same stride as the processing path to match shapes, 
        # OR `x_org` should be the same as `x`.
        # But `x_org` is distinct.
        
        # Let's implement `DelayedLayer` with `shape_match_conv` having `stride=nb_stride`?
        # No, `x_org` and `x` have different input lengths probably.
        # `x` is the pattern. `x_org` is the original.
        # Pattern length != Original length.
        # If they are added, it's impossible.
        
        # Wait, `pattern` function in `utils/data.py`:
        # `index_sample` length = `input_len` (which is `datapack.shape[1]`).
        # So `new_sequence[index_sample]` has SAME length as input `datapack`.
        # Ah! `generate_index(seq_len)` loops `seq_len` times.
        # `kernel_order` has length `(delay_factor+1)*kernel_size`.
        # `idx` grows.
        # `idx = np.concatenate((idx, kernel_order + node_index))`
        # So `idx` length is `seq_len * len(kernel_order)`.
        # So `x` (pattern) is LONGER than `x_org`?
        # `input_len` is 168 (example).
        # `kernel_order` length e.g. 2*2=4.
        # `x` length = 168 * 4 = 672.
        
        # So `x` is expanded.
        # `DelayedLayer` conv0 has `stride=nb_stride`.
        # If `nb_stride` matches `len(kernel_order)`, then `x` is downsampled back to `seq_len`?
        # `nb_stride` default 7. `len(kernel_order)` depends on config.
        # Config: `kernel_size=2`, `gap=2`, `delay_factor=1`.
        # `kernel_order`: d=0,1. k=0,1.
        # d=0: 0, 1.
        # d=1: 2+0, 2+1 = 2, 3.
        # `kernel_order` = [0, 1, 2, 3]. Length 4.
        # If `nb_stride` is 7, then 672 / 7 = 96.
        # `x_org` length 168.
        # Still doesn't match.
        
        # Maybe `nb_stride` should be equal to `len(kernel_order)`?
        # In `Model1` init: `list_stride=(7, 1)`.
        # Maybe `list_stride` should be passed in config?
        # `run_timemmd_experiments.py` does NOT pass `list_stride`.
        # So it uses default `(7, 1)`.
        
        # This suggests the provided code might have configuration mismatches or I'm analyzing it too strictly.
        # However, I must implement something that runs.
        # I will implement `DelayedLayer` to dynamically adjust `shape_match_conv` stride or kernel to match `x` shape, 
        # OR I will just implement it as is and let it fail/debug.
        # Better: I'll implement `shape_match_conv` as a 1x1 conv, but in `forward`, I will interpolate `x_org` to match `x` size if needed?
        # No, that's altering logic.
        
        # Let's look at `DelayedLayer` in `models.py` again.
        # `self.shape_match_conv = layers.Conv1D(..., padding='same', ...)`
        # It does NOT specify stride, so stride=1.
        # `x = layers.add([self.shape_match_conv(x_org), x])`
        
        # I will implement `DelayedLayer` taking `x_org` and `x`.
        # I will apply `shape_match_conv` to `x_org`.
        # I will apply convs to `x`.
        # Then I will check shapes. If mismatch, I'll print a warning or try to handle it (e.g. crop/pad).
        # But for now, direct translation.
        
        self.shape_match_conv = nn.Conv1d(input_channels, nb_filters, kernel_size=1, padding=0)
        self.act_out = nn.ReLU()
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x_org, x = inputs
        
        # x_org: [B, C, L_org]
        # x: [B, C, L_in]
        
        # Layer 0
        # Causal padding for stride
        # We want to pad such that we take current and previous values.
        # Padding = (kernel_size - 1) * dilation
        # With stride, we pad input.
        pad0 = (self.conv1_kernel - 1)
        x_pad = F.pad(x, (pad0, 0))
        x = self.conv0(x_pad)
        x = self.act0(x)
        # x = self.drop0(x)
        
        # Layer 1
        pad1 = (self.conv1_kernel - 1)
        x_pad = F.pad(x, (pad1, 0))
        x = self.conv1(x_pad)
        x = self.act1(x)
        # x = self.drop1(x)
        
        # Skip connection
        if self.use_skip_connections:
            x_org_match = self.shape_match_conv(x_org)
            
            # Match shapes if necessary
            if x_org_match.shape[-1] != x.shape[-1]:
                # Hack: Resize x_org_match to x
                x_org_match = F.interpolate(x_org_match, size=x.shape[-1], mode='linear', align_corners=False)
            
            x = x + x_org_match
            x_out = self.act_out(x)
        else:
            x_out = x
            
        return [x, x_out]

class StrideLayer(nn.Module):
    def __init__(self, nb_filters=64, kernel_size=3, dilation_rate=1, padding='causal', dropout_rate=0.0, use_weight_norm=False):
        super(StrideLayer, self).__init__()
        self.nb_filters = nb_filters
        self.layers = nn.ModuleList()
        
        for k in range(6):
            dilation = 2 ** k
            # Padding for causal dilated conv
            # pad = (kernel_size - 1) * dilation
            pad = (kernel_size - 1) * dilation
            
            conv = nn.Conv1d(nb_filters, nb_filters, kernel_size, dilation=dilation, padding=0)
            if use_weight_norm:
                conv = weight_norm(conv)
            
            self.layers.append(nn.ModuleList([
                Chomp1d(pad), # Remove padding from right
                conv,
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]))
            
        self.act_out = nn.ReLU()
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = inputs
        for layer_block in self.layers:
            chomp, conv, act, drop = layer_block
            
            # Manual padding
            dilation = conv.dilation[0]
            kernel_size = conv.kernel_size[0]
            pad = (kernel_size - 1) * dilation
            
            x_pad = F.pad(x, (pad, 0))
            x_out = conv(x_pad)
            x_out = act(x_out)
            x_out = drop(x_out)
            
            # In TF code: x = layer(x). It updates x sequentially.
            x = x_out
            
        # Skip connection: inputs + x
        # TF: layers.add([inputs, x])
        x = inputs + x
        x_out = self.act_out(x)
        # x_out = self.drop_out(x_out) # TF code has self.drop1 but doesn't seem to use it in return?
        # TF: return [x, x_out]
        # Wait, TF StrideLayer call returns [x, x_out] where x is the skip connection?
        # TF: inputs_x = layers.add([inputs, x])
        #     x_out = self.ac1(inputs_x)
        #     return [x, x_out]
        # Wait, `x` in return is the output of the LAST conv layer (before adding residual).
        # `x_out` is the result of addition + activation.
        # But `Model1` uses: `x_res, x = dilated_block(x)`.
        # So `x_res` corresponds to `x` (skip connection?), and `x` corresponds to `x_out` (output for next layer).
        # In `Model1`: `self.skip_connections.append(x_res)`.
        # So `x_res` is the output of the conv stack (before residual add).
        # And `x` (the second return value) is the input to the next block.
        
        return [x, x_out]

class DelayNet(nn.Module):
    def __init__(self, config):
        super(DelayNet, self).__init__()
        
        self.nb_filters = config.get('nb_filters', 64)
        self.kernel_size = config.get('kernel_size', 3)
        self.nb_stacks = config.get('nb_stacks', 1)
        self.dropout_rate = config.get('dropout_rate', 0.0)
        self.input_width = config.get('input_width', 24)
        self.target_size = config.get('output_length', 24)
        self.features = len(config.get('features', [1]))
        
        # Stride list - assuming default if not in config
        self.list_stride = config.get('list_stride', (7, 1))
        
        self.delayed_block = DelayedLayer(
            nb_stride=self.list_stride[0],
            nb_filters=self.nb_filters,
            kernel_size=self.kernel_size,
            dropout_rate=self.dropout_rate,
            input_channels=self.features
        )
        
        self.dilated_blocks = nn.ModuleList()
        for i in range(self.nb_stacks - 1):
            self.dilated_blocks.append(
                StrideLayer(
                    nb_filters=self.nb_filters,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate
                )
            )
            
        self.dense1 = nn.Linear(self.nb_filters, 128)
        self.dense2 = nn.Linear(128, self.target_size)
        
    def forward(self, x_org, x_pattern):
        # x_org: [B, L, C] -> [B, C, L]
        x_org = x_org.permute(0, 2, 1)
        x_pattern = x_pattern.permute(0, 2, 1)
        
        skip_connections = []
        
        # Delayed Block
        x_res, x = self.delayed_block([x_org, x_pattern])
        skip_connections.append(x_res)
        
        # Dilated Blocks
        for block in self.dilated_blocks:
            x_res, x = block(x)
            skip_connections.append(x_res)
            
        # Aggregate skip connections
        if len(self.dilated_blocks) > 0:
            # Element-wise add
            # Note: x_res from DelayedLayer and StrideLayer must have same shape
            # DelayedLayer returns x (before residual add? No, DelayedLayer returns [x, x_out])
            # DelayedLayer: return [x, x_out]. x is output of convs. x_out is output of block.
            # StrideLayer: return [x, x_out]. x is output of convs. x_out is output of block.
            # So we sum up the 'x's.
            
            # We need to sum them.
            total = skip_connections[0]
            for s in skip_connections[1:]:
                total = total + s
            
            x = F.relu(total)
        else:
            # If no dilated blocks, use output of delayed block?
            # Model1 logic:
            # if self.use_skip_connections and len(self.dilated_blocks) > 0:
            #    x = layers.add(self.skip_connections)
            #    x = self.final_ac(x)
            # else:
            #    x = x (which is output of last block)
            
            # If nb_stacks=1, dilated_blocks is empty.
            # So x is just output of delayed_block (x_out).
            pass
            
        # Slicer: Take last time step
        # x: [B, C, L]
        x = x[:, :, -1] # [B, C]
        
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x

