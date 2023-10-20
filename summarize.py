import ray
from ray.tune import ExperimentAnalysis
import numpy as np
from os import path
import os
import shutil
import json


def scatter_plot_columns(df, x, y, file_name, c=None):
    axs = df.plot.scatter(x=x, y=y, c=c)
    fig = axs.get_figure()
    fig.savefig(file_name+".png")

def hist_plot(df, x, file_name):
    axs = df[x].plot.hist()
    fig = axs.get_figure()
    fig.savefig(file_name+".png")

# CAUTION:
# If config copying is on, then the exp_list can only be for one set of experiments i.e. all algorithms for one particular setting
# AND the exp_names list must match the exp_list.
# Edit the output_dir to be for the setting in question. This is where the optimal configs will be copied to, named according to the exp_names list.
# Any new parameters w.r.t. which we tune must be added to the tune_params list for proper printing.

exp_list = [
"ASHA_Time_tb_ASHA_cheap_uci_energy_Adam_Adam",
"ASHA_Time_tb_ASHA_cheap_uci_energy_KFACDeepMind_KFACDeepMind",
"ASHA_Time_tb_ASHA_cheap_uci_energy_KFACKazuki_KFACKazuki",
"ASHA_Time_tb_ASHA_cheap_uci_energy_OursAdaptive_OursAdaptive",
"ASHA_Time_tb_ASHA_cheap_uci_energy_OursDampedUnadaptive_OursDampedUnadaptive",
"ASHA_Time_tb_ASHA_cheap_uci_energy_SGD_SGD",
    ]

# EDIT ME
output_dir = "./results/uci_energy_cheap"

copy_optimal_configs = True

tuning_metric = 'validation_loss'

tune_params = [
        "opt.initial_damping", 
        "shanks_acceleration_dict:num_update_steps", 
        "shanks_acceleration_dict:acceleration_order", 
        "opt.eps", 
        "adam_one_minus_b1",
        "adam_one_minus_b2",
        "opt.learning_rate", 
        "opt.momentum", 
        "root.batch_size"
]

exp_names = [
"Adam",
"KFACDeepMind",
"KFACKazuki",
"OursAdaptive",
"OursDampedUnadaptive",
"SGD",
]

if copy_optimal_configs:
    assert len(exp_names) == len(exp_list)



if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def print_tracked_tune_parameters(config_dict, params):
    found_params_string, params_string = "", ""
    # For each parameter, check if it exists in the config dict
    for p in params:
        # If the parameter is inside another dict, extract the parent/child keys
        if p.find(":") != -1:
            subp = p.split(":")
            parent_key, child_key = subp[0], subp[1]
            if parent_key in config_dict and child_key in config_dict[parent_key]:
                value = config_dict[parent_key][child_key]
            else:
                continue
        else:
            value = config_dict.get(p, None)
            if value is None:
                continue
        found_params_string += p + ","
        params_string += "{:.9},".format(float(value))
    print(found_params_string)
    print(params_string)

def print_losses(training_loss, test_loss, validation_loss):
    print("training_loss,test_loss,validation_loss,")
    print("{:.6f},{:.6f},{:.6f}".format(training_loss, test_loss, validation_loss))

def copy_best_config(exp_path, exp_name, best_config):
    tensorboard_path_start = exp_path.index("tb_")
    tensorboard_path_end = exp_path.index(exp_name) + len(exp_name)
    tensorboard_path = exp_path[tensorboard_path_start:tensorboard_path_end]
    full_tensorboard_path = os.path.join("./runs", tensorboard_path)
    all_subfolders = [ f.path for f in os.scandir(full_tensorboard_path) if f.is_dir() ]
    matching_subfolders = []
    matching_configs = []

    for sf in all_subfolders:
        with open(os.path.join(sf, "config.json")) as f:
            trial_config = json.load(f)
            # Try to match the config with our best trial config using lr or initial damping
            # They're drawn from a contiuous distrib, so good chance of uniqueness
            test_key = "learning_rate"
            if not (test_key in trial_config["optimiser"]):
                test_key = "initial_damping"
            if trial_config["optimiser"][test_key] == best_config["opt."+test_key]:
                matching_subfolders.append(sf)
                matching_configs.append(trial_config)


    if len(matching_subfolders) == 0:
        print("ERROR: Failed to find matching config file for {}".format(exp_name))
        return
    elif len(matching_subfolders) == 1:
        found_trial_dir = matching_subfolders[0]
    else:
        print("Please select correct directory to match timestamp {}".format(best_config["trial_name"]))
        for j, subfolder in enumerate(matching_subfolders):
            print("{}. {}".format(j, subfolder))
            print("{}".format(matching_configs[j]))
        f_index = -1
        while True:
            option = input("prompt")
            try:
                f_index = int(option)
            except ValueError:
                print("Not a valid integer")
            if f_index >= 0 and f_index < len(matching_subfolders):
                break
        found_trial_dir = matching_subfolders[f_index]

    best_config_file = os.path.join(found_trial_dir, "config.json")
    shutil.copyfile(best_config_file, os.path.join(output_dir, "{}.json".format(exp_name)))


for i, exp_path in enumerate(exp_list):
    print(exp_path)
    analysis = ExperimentAnalysis(path.join("./runs",exp_path))
    trial_dfs = analysis.trial_dataframes
    best_dict = {
        'validation_loss': np.Infinity,
        'training_loss': np.Infinity,
        'test_loss': np.Infinity,
        'trial_key': None,
    }

    for key in trial_dfs.keys():
        best_row = None
        if trial_dfs[key]["training_loss"].isnull().values.any() or trial_dfs[key]["validation_loss"].isnull().values.any():
            # Get last good loss
            df = trial_dfs[key]
            not_nan_df = df[~df[tuning_metric].isnull()]
            if len(not_nan_df) == 0:
                continue
            best_row = not_nan_df.iloc[-1] # We don't expect a NaN validation loss to become not-NaN later in the run. So take the last non-null row
            assert not np.isnan(best_row["training_loss"])
            #assert not np.isnan(best_row["validation_loss"])
        else:
            best_row = trial_dfs[key].iloc[-1] # Get last row

        if best_row[tuning_metric] < best_dict[tuning_metric]:
            best_dict['validation_loss'] = best_row["validation_loss"]
            best_dict["trial_key"] = key
            best_dict["training_loss"] = best_row["training_loss"]
            best_dict["test_loss"] = best_row["test_loss"]

    # For comparison, selecting best ignoring NaNs would choose:
    best_config = analysis.get_all_configs()[best_dict["trial_key"]]

    f = open(path.join(output_dir, "summary.txt"), "a")
    f.write(exp_path + "\n")
    f.write("training_loss: {:f}\n".format(best_dict["training_loss"]))
    f.write("test_loss: {:f}\n".format(best_dict["test_loss"]))
    f.write("validation_loss: {:f}\n".format(best_dict['validation_loss']))
    f.write(str(best_config)+"\n")
    f.close()
    print("--------------------------------")
    print(exp_path)
    print("Tuning with respect to {}".format(tuning_metric))
    #print("training_loss: {:f}".format(best_dict["training_loss"]))
    #print("test_loss: {:f}".format(best_dict["test_loss"]))
    #print("validation_loss: {:f}".format(best_dict['validation_loss']))
    #print(str(best_config))

    print_losses(best_dict["training_loss"], best_dict["test_loss"], best_dict['validation_loss'])
    print_tracked_tune_parameters(best_config, tune_params)
    if copy_optimal_configs:
        copy_best_config(exp_path, exp_names[i], best_config)
    print("--------------------------------")

"""
    df = analysis.dataframe(tuning_metric, "min")
    best_nonnan_trial = analysis.get_best_trial(tuning_metric, "min")
    best_nonnan_row = best_nonnan_trial.last_result
    print("***Non-Nan Trial for Comparison**")
    print(str(best_nonnan_trial.evaluated_params))
    print("training_loss: {:f}".format(best_nonnan_row["training_loss"]))
    print("test_loss: {:f}".format(best_nonnan_row["test_loss"]))
    print("validation_loss: {:f}".format(best_nonnan_row["validation_loss"]))
    print(str(best_nonnan_trial.config))
"""
