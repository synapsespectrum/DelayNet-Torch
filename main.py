#  Copyright (c) 2022-2022 Andrew
#  Email: andrewlee1807@gmail.com
#
import argparse
import os

from models import get_model, build_callbacks
from utils.data import Dataset, TimeSeriesGenerator
from utils.logging import arg_parse, warming_up, close_logging


def main():
    parser = argparse.ArgumentParser()
    args = arg_parse(parser)

    config = warming_up(args)

    # Load dataset
    if config["dataset_name"] == "None":
        if args.dataset_path is not None and args.dataset_path.split(".")[-1] in ["csv", "CSV"]:
            if args.features is None or args.prediction_feature is None:
                print("Please enter the features to use for training")
                exit(0)
            import pandas as pd
            df = pd.read_csv(args.dataset_path, sep='\t')
            data = df[args.features.split(',')].to_numpy() # (Sequence, Number of features)
            config["prediction_feature"] = args.prediction_feature
        else:
            print("Please enter the dataset path as csv file")
            exit(0)
    else:
        dataset = Dataset(dataset_name=config["dataset_name"],
                          noise_std=config.get("noise_std", 0.0),
                          smoothing_window=config.get("smoothing_window", 0))

        data = dataset.dataloader.export_the_sequence(config["features"])

    print("Building time series generator...")
    tsf = TimeSeriesGenerator(data=data,
                              config=config,
                              normalize_type=1,
                              shuffle=False)

    if config.get("model_name") != "GRU":
        tsf.re_arrange_sequence(config)
        data_train = [[tsf.data_train[0], tsf.data_train_gen[0]], tsf.data_train[1]]
        data_valid = [[tsf.data_valid[0], tsf.data_valid_gen[0]], tsf.data_valid[1]]
        data_test = [[tsf.data_test[0], tsf.data_test_gen[0]], tsf.data_test[1]]
    else:
        data_train = [tsf.data_train[0], tsf.data_train[1]]
        data_valid = [tsf.data_valid[0], tsf.data_valid[1]]
        data_test = [tsf.data_test[0], tsf.data_test[1]] if tsf.data_test is not None else None

    if config.get("channel_independence", False):
        print("Applying Channel Independence...")
        def reshape_ci(data_list):
            # data_list is [X, Y] or [[X_org, X_pattern], Y]
            if isinstance(data_list[0], list): # [[X_org, X_pattern], Y]
                X_org = data_list[0][0]
                X_pattern = data_list[0][1]
                Y = data_list[1]
                
                # Reshape X_org: (B, S, C) -> (B*C, S, 1)
                B, S, C = X_org.shape
                X_org = X_org.transpose(0, 2, 1).reshape(B * C, S, 1)
                
                # Reshape X_pattern: (B, P, C) -> (B*C, P, 1)
                B, P, C = X_pattern.shape
                X_pattern = X_pattern.transpose(0, 2, 1).reshape(B * C, P, 1)
                
                # Reshape Y: (B, O, C) -> (B*C, O, 1)
                B, O, C = Y.shape
                Y = Y.transpose(0, 2, 1).reshape(B * C, O, 1)
                
                return [[X_org, X_pattern], Y]
            else: # [X, Y] (for GRU etc)
                X = data_list[0]
                Y = data_list[1]
                
                # Reshape X: (B, S, C) -> (B*C, S, 1)
                B, S, C = X.shape
                X = X.transpose(0, 2, 1).reshape(B * C, S, 1)
                
                # Reshape Y: (B, O, C) -> (B*C, O, 1)
                B, O, C = Y.shape
                Y = Y.transpose(0, 2, 1).reshape(B * C, O, 1)
                
                return [X, Y]

        data_train = reshape_ci(data_train)
        data_valid = reshape_ci(data_valid)
        if data_test is not None:
            data_test = reshape_ci(data_test)
            
        print("DEBUG: data_train shapes:")
        if isinstance(data_train[0], list):
            print("X_org:", data_train[0][0].shape)
            print("X_pattern:", data_train[0][1].shape)
        else:
            print("X:", data_train[0].shape)
        print("Y:", data_train[1].shape)
            
        # Update config input_width/features for model build if necessary
        # The model will see 1 feature now
        # But we shouldn't change config['features'] list because it might be used elsewhere
        # However, the model build uses len(config['features'])
        # We need to trick the model to think there is only 1 feature
        config['original_features'] = config['features']
        config['features'] = ['channel_independent_feature']

    print("Building model...")
    # Get model (built and summary)
    model = get_model(config=config)

    # callbacks
    model_filepath = os.path.join(config["output_dir"], f"{config['dataset_name']}_{config['output_length']}_best_weights.ckpt")
    callbacks = build_callbacks(tensorboard_log_dir=config["tensorboard_log_dir"], filepath=model_filepath)

    # Train model
    history = model.fit(x=data_train[0],  # [number_recoder, input_len, number_feature]
                        y=data_train[1],  # [number_recoder, output_len, number_feature]
                        validation_data=data_valid,
                        epochs=config["epochs"],
                        callbacks=[callbacks],
                        verbose=2,
                        batch_size=64,
                        use_multiprocessing=True)

    print("=============================================================")
    print("Minimum val mse:")
    print(min(history.history['val_mse']))
    print("Minimum training mse:")
    print(min(history.history['mse']))
    result = model.evaluate(data_test[0], data_test[1], batch_size=1,
                            verbose=2,
                            use_multiprocessing=True)
    print("Evaluation result: ")
    print(config['output_length'], result[1], result[2])

    result_file = f'{os.path.join(config["output_dir"], config["dataset_name"])}_evaluation_result.txt'
    file = open(result_file, 'a')
    file.write(f'{config["output_length"]},{result[1]},{result[2]}\n')
    file.close()

    if args.write_log_file:
        close_logging(config["file"], config["orig_stdout"])


if __name__ == '__main__':
    main()
