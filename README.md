# DelayNet-PyTorch-Multivariate

PyTorch reimplementation of **DelayNet** for **multivariate time series forecasting**.

Original paper:  
[DelayNet: Enhancing Temporal Feature Extraction for Electronic Consumption Forecasting with Delayed Dilated Convolution](https://www.mdpi.com/1996-1073/16/22/7662)  
Original repo (TensorFlow, mainly univariate): https://github.com/andrewlee1807/DelayNet

## Key Changes
- Framework: PyTorch (instead of TensorFlow)
- Supports **multivariate** time series (multiple input channels/features)
- Clean, modular implementation with configurable delayed dilated convolutions
- Flexible data loading for custom multivariate datasets

## Main Features
- Delayed dilated convolution blocks (core innovation of DelayNet)
- Configurable: kernel_size, gap, delay_factor, nb_filters, nb_stacks, etc.
- Suitable for long-range time series forecasting tasks

## Installation
```bash
pip install -r requirements.txt
# or use your preferred environment (torch, pandas, numpy, etc.)
