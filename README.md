# DelayNet-Multivariate

This is a standalone project extracted from the original DelayNet repository, specifically tailored to run multivariable time series forecasting using the Shared Weights mechanism (`channel_independence: True`).

Original paper:  
[DelayNet: Enhancing Temporal Feature Extraction for Electronic Consumption Forecasting with Delayed Dilated Convolution](https://www.mdpi.com/1996-1073/16/22/7662)  
Original repo (TensorFlow, mainly univariate): https://github.com/andrewlee1807/DelayNet

## Environment Setup

The required packages and dependencies are preserved from the `ts_model` conda environment. 

1. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate delaynet
   ```

*(Alternatively, if `environment.yml` behaves unexpectedly across different platforms, ensure you have Python, TensorFlow/Keras, Pandas, Numpy, and Scikit-Learn installed.)*

## Running an Experiment

The configurations for multivariate datasets should reside in `benchmark/config/`. The core flag to trigger shared weights multivariate forecasting is:
```yaml
channel_independence: True
```

To run the experiment, use `main.py` directly with your configuration file, or use the evaluation wrap script:

```bash
# Example running the multivariate CNU test
python run_experiments.py
```

### Dataset Structure
Ensure that your multivariable datasets are placed inside the `dataset/` folder. The corresponding `yaml` configuration needs to list the target features in `features: ["feature1", "feature2", ...]`.

### Important Scripts
- `main.py`: The entry point for building and training the network. Includes explicit reshaping for `channel_independence`.
- `models.py` / `delayedtcn/models.py`: Architecture definitions including the `DelayLayer` and `DilatedBlock` optimized for multivariate series.
- `run_experiments.py`: Hardcoded script to specifically run the multivariate test.
