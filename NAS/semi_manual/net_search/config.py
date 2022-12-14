import yaml

def read_config(filename):
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
        return config


if __name__  == "__main__":
    # create sample config

    config = {
        "epochs": 10,
        "criterion": "CrossEntropyLoss",
        "optimizer": "Adam",
        "optimizer_kwargs":  {
            "lr": 1e-04
        }
    }

    with open("train_cfg_example.yaml", "w") as f:
        f.write(yaml.dump(config))

        
    
