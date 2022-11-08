from os.path import join

import numpy as np

from pacml.experiment import start_hp_search
from pac.general import hash_from_dict
from pac.write import create_dir
from training.lightning_trainer.train_pipeline import (
    run,
    doConnection,
    getLabelEncoder,
    getTrainValFilesLists,
    callbacks,
)
from pacml.models.torch.callbacks import PACLogger

CONFIG_INPUT_PATH = "femke_config.yaml"
DEBUG = True


def my_run_wrapper(experiment):
    config = experiment.config
    
    # use smaller dataset when debugging
    if DEBUG:
        config["PATH_TRAIN_VAL_DATASET"] = "/data/phd_femke/soundsolution/Ben_backup/Model_training_data_debug/"

    # Do the connection
    myfs = doConnection(config["CONNECTION_STRING"])

    # Write the labels
    getLabelEncoder(config, myfs)

    # Get the train & val file list, either from cache, or by processing
    cache_identifier_dict = {
        "path_train_val_dataset": config["PATH_TRAIN_VAL_DATASET"], 
        "prop_training": config["PROP_TRAINING"], 
        "length_segments": config["LENGTH_SEGMENTS"],
        "sample_rate": config["SAMPLE_RATE"],
        "debug": DEBUG,
        }
    cache_identifier = hash_from_dict(cache_identifier_dict)
    try:
        train_list = np.load(join("temp_data_cache", cache_identifier + "_train_list.npy"), allow_pickle=True)
        val_list = np.load(join("temp_data_cache", cache_identifier + "_val_list.npy"), allow_pickle=True)
    except:
        train_list, val_list = getTrainValFilesLists(config, myfs)
        create_dir("temp_data_cache")
        np.save(join("temp_data_cache", cache_identifier + "_train_list.npy"), train_list, allow_pickle=True)
        np.save(join("temp_data_cache", cache_identifier + "_val_list.npy"), val_list, allow_pickle=True)
    
    # Get callbacks
    cbacks = callbacks(config)

    # Start run from original code
    run(
        experiment.config,
        list_train=train_list,
        list_val=val_list,
        callbacks=cbacks,
        logger=PACLogger(experiment),
    )


if __name__ == "__main__":
    # do not ask for tag when debugging
    if not DEBUG:
        tag_prefix = None
    else:
        tag_prefix = ""

    # Start the search
    start_hp_search(
        CONFIG_INPUT_PATH,
        my_run_wrapper,
        tag_prefix=tag_prefix,
        output_dir="experiment_output",
    )
