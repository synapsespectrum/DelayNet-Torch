
import os
import subprocess
import time

# Experiment configurations
output_lengths = [1, 24, 168]
configs = [
    {
        "name": "Multivariate",
        "config_path": "benchmark/config/cnu/cnu_multivariate.yaml",
        "output_dir": "results/cnu_multivariate_comparison",
        "dataset_name": "CNU_ENGINEERING_7",
        "features": "temperatures,energy", # Will be handled by config file, but good to note
        "prediction_feature": "temperatures,energy"
    }
]

def run_experiment(config, length):
    print(f"Running {config['name']} experiment with output_length={length}...")
    
    cmd = [
        "python", "main.py",
        "--config_path", config["config_path"],
        "--output_dir", config["output_dir"],
        "--dataset_name", config["dataset_name"],
        "--output_length", str(length)
    ]
    
    # For Univariate, we need to ensure features are correct if overridden by command line (though config file handles it)
    # But main.py logic for features override is a bit tricky if config file has list.
    # Let's rely on the config file for features, but we need to make sure cnu_delay1.yaml is correct.
    # cnu_delay1.yaml has features: ["PowerConsumption"].
    # cnu_multivariate.yaml has features: ["temperatures", "energy"].
    
    # However, main.py might override if we pass --features. 
    # Let's NOT pass --features and rely on config files being correct.
    # But wait, cnu_multivariate.yaml has channel_independence: True.
    # cnu_delay1.yaml does NOT.
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished {config['name']} experiment with output_length={length}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {config['name']} experiment with output_length={length}: {e}")

def main():
    for length in output_lengths:
        for config in configs:
            run_experiment(config, length)
            time.sleep(2) # Cool down

if __name__ == "__main__":
    main()
