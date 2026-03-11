import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
from utils.data import Dataset, TimeSeriesGenerator
from models import get_model

def load_and_predict(config_path, output_dir, dataset_name, output_length=1):
    """Load configuration, prepare data, build model, and predict the multivariate test set."""
    
    with open(config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    config["output_length"] = output_length
    config["dataset_name"] = dataset_name
    config["output_dir"] = output_dir
    
    print(f"Loading dataset: {dataset_name}")
    dataset = Dataset(dataset_name=config["dataset_name"],
                      noise_std=config.get("noise_std", 0.0),
                      smoothing_window=config.get("smoothing_window", 0))
    
    data = dataset.dataloader.export_the_sequence(config["features"])
    
    print("Building time series generator...")
    tsf = TimeSeriesGenerator(data=data,
                              config=config,
                              normalize_type=1,
                              shuffle=False)
    
    if config.get("model_name", "DelayNet") != "GRU":
        tsf.re_arrange_sequence(config)
        data_test = [[tsf.data_test[0], tsf.data_test_gen[0]], tsf.data_test[1]]
    else:
        data_test = [tsf.data_test[0], tsf.data_test[1]]
    
    if config.get("channel_independence", False):
        print("Applying Channel Independence on Test Sequence...")
        def reshape_ci(data_list):
            if isinstance(data_list[0], list):
                X_org = data_list[0][0]
                X_pattern = data_list[0][1]
                Y = data_list[1]
                
                B, S, C = X_org.shape
                X_org = X_org.transpose(0, 2, 1).reshape(B * C, S, 1)
                
                B, P, C = X_pattern.shape
                X_pattern = X_pattern.transpose(0, 2, 1).reshape(B * C, P, 1)
                
                B, O, C = Y.shape
                Y = Y.transpose(0, 2, 1).reshape(B * C, O, 1)
                
                return [[X_org, X_pattern], Y]
            else:
                X = data_list[0]
                Y = data_list[1]
                
                B, S, C = X.shape
                X = X.transpose(0, 2, 1).reshape(B * C, S, 1)
                
                B, O, C = Y.shape
                Y = Y.transpose(0, 2, 1).reshape(B * C, O, 1)
                
                return [X, Y]
        
        data_test = reshape_ci(data_test)
        
        config['original_features'] = config['features']
        config['features'] = ['channel_independent_feature']
    
    print("Building model...")
    model = get_model(config=config)
    
    weights_path = os.path.join(config["output_dir"], f"{config['dataset_name']}_{config['output_length']}_best_weights.ckpt")
    if os.path.exists(weights_path + ".index") or os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_weights(weights_path)
    else:
        print(f"Warning: No weights found at {weights_path}")
    
    print("Predicting on test set...")
    y_pred = model.predict(data_test[0], batch_size=1)
    y_true = data_test[1]
    
    if tsf.scaler_engine is not None:
        if config.get("channel_independence", False):
            num_channels = len(config.get('original_features', config['features']))
            num_samples = y_pred.shape[0] // num_channels
            
            y_pred_reshaped = y_pred.reshape(num_samples, num_channels, -1)
            y_true_reshaped = y_true.reshape(num_samples, num_channels, -1)
            
            y_pred_for_inverse = y_pred_reshaped.transpose(0, 2, 1).reshape(-1, num_channels)
            y_true_for_inverse = y_true_reshaped.transpose(0, 2, 1).reshape(-1, num_channels)
            
            y_pred_inverse = tsf.scaler_engine.inverse_transform(y_pred_for_inverse)
            y_true_inverse = tsf.scaler_engine.inverse_transform(y_true_for_inverse)
            
            y_pred = y_pred_inverse.reshape(num_samples, output_length, num_channels)
            y_true = y_true_inverse.reshape(num_samples, output_length, num_channels)
        else:
            y_pred_flat = y_pred.flatten().reshape(-1, 1)
            y_true_flat = y_true.flatten().reshape(-1, 1)
            
            y_pred_inverse = tsf.scaler_engine.inverse_transform(y_pred_flat)
            y_true_inverse = tsf.scaler_engine.inverse_transform(y_true_flat)
            
            y_pred = y_pred_inverse.reshape(-1, output_length)
            y_true = y_true_inverse.reshape(-1, output_length)
    
    return y_pred, y_true, config

def visualize_multivariate(args):
    """Visualize multivariate forecasting results separately."""
    print("=" * 60)
    print(f"Executing Multivariate Visualization... (Output length={args.output_length})")
    print("=" * 60)
    
    y_pred_multi, y_true_multi, config_multi = load_and_predict(
        config_path=args.config_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        output_length=args.output_length
    )
    
    print("=" * 60)
    print("Creating Visualization Plot...")
    print("=" * 60)
    
    num_samples = min(5, y_pred_multi.shape[0])
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        if len(y_true_multi.shape) == 3:  # (B, O, C)
            for c in range(y_true_multi.shape[2]):
                feature_name = config_multi.get('original_features', config_multi['features'])[c]
                ax.plot(y_true_multi[i, :, c], label=f'GT - {feature_name}', linewidth=2, alpha=0.8)
                ax.plot(y_pred_multi[i, :, c], label=f'Pred - {feature_name}', linewidth=2, linestyle='--', alpha=0.8)
        else:
            ax.plot(y_true_multi[i], label='Ground Truth', linewidth=2, alpha=0.8)
            ax.plot(y_pred_multi[i], label='Multivariate Prediction', linewidth=2, linestyle='--', alpha=0.8)
            
        ax.set_title(f'Sample {i+1} - Multivariate Forecast', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step (hours)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if not os.path.exists('results'):
        os.makedirs('results')
        
    output_path = f"results/multivariate_forecast_{args.output_length}h.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization Plot successfully saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multivariate Forecasting Visualization")
    parser.add_argument('--config_path', type=str, default='benchmark/config/cnu/cnu_multivariate.yaml')
    parser.add_argument('--output_dir', type=str, default='results/cnu_multivariate_comparison')
    parser.add_argument('--dataset_name', type=str, default='CNU_ENGINEERING_7')
    parser.add_argument('--output_length', type=int, default=1)
    
    args = parser.parse_args()
    visualize_multivariate(args)
