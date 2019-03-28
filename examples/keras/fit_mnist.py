
# coding: utf-8

# ## Imports

# In[ ]:


import sys
import os
sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0])
config_filepath = os.path.join(os.getcwd(),"config/fit_config_mnist.json")
notebook_filepath = os.path.join(os.getcwd(),"fit_mnist.ipynb")
import uuid
import json
import datetime
import getpass

from mercury_ml.common import tasks
from mercury_ml.common import utils
from mercury_ml.common import containers as common_containers
from mercury_ml.keras import containers as keras_containers


# In[ ]:


session_id = str(uuid.uuid4().hex)


# ### load config file and update placeholder

# In[ ]:


config = utils.load_referenced_json_config(config_filepath)


# In[ ]:


utils.recursively_update_config(config["meta_info"], {
    "session_id": session_id,
    "model_purpose": config["meta_info"]["model_purpose"],
    "config_filepath": config_filepath,
    "notebook_filepath": notebook_filepath
})


# In[ ]:


utils.recursively_update_config(config, config["meta_info"])


# ### Helper functions
# `get_and_log`, or basically `getattr`, is used to read out function names from the config file.
# This is necesarry because different functions are used in different context. E.g. there are different functions for Keras than for H2O. Or for reading images instead of reading arrays from .csv.

# In[ ]:


def create_and_log(container, class_name, params):
    provider = getattr(container, class_name)(**params)
    print("{}.{}".format(container.__name__, class_name))
    print("params: ", json.dumps(params, indent=2))
    return provider

def get_and_log(container, function_name):
    provider = getattr(container, function_name)
    print("{}.{}".format(container.__name__, function_name))
    return provider
def maybe_transform(data_bunch, pre_execution_parameters):
    if pre_execution_parameters:
        return data_bunch.transform(**pre_execution_parameters)
    else:
        return data_bunch


# ### Download and convert MNIST Images
# If the images for training do not exist, download them and save them as .png images

# In[ ]:


if not os.path.exists(config["exec"]["read_source_data"]["params"]["train_params"]["iterator_params"]["directory"]):
    from load_mnist import download_and_convert_mnist 
    download_and_convert_mnist(config["meta_info"]["data_bunch_name"])


# ## Source
# First read out from the config file which file input
# Second read the data from source into a generator, ready for training

# In[ ]:


read_source_data_set = get_and_log(common_containers.SourceReaders, config["init"]["read_source_data"]["name"])


# In[ ]:


data_bunch_source = tasks.read_train_valid_test_data_bunch(read_source_data_set,**config["exec"]["read_source_data"]["params"] )


# If there are parameters speifed for pre_execution_transformation, the data has to be transformed

# In[ ]:


data_bunch_fit = maybe_transform(data_bunch_source, config["exec"]["fit"].get("pre_execution_transformation"))


# ## Model

# Load keras functions from config file

# In[ ]:


get_optimizer = get_and_log(keras_containers.OptimizerFetchers, 
                           config["init"]["get_optimizer"]["name"])
get_loss_function = get_and_log(keras_containers.LossFunctionFetchers, 
                                config["init"]["get_loss_function"]["name"])
compile_model = get_and_log(keras_containers.ModelCompilers, 
                            config["init"]["compile_model"]["name"])
fit = get_and_log(keras_containers.ModelFitters, config["init"]["fit"]["name"])


# Specify the keras model

# In[ ]:


if(config["exec"]["read_source_data"]["params"]["train_params"]["iterator_params"]["color_mode"]=="grayscale"):
    color_size = 1
elif(config["exec"]["read_source_data"]["params"]["train_params"]["iterator_params"]["color_mode"]=="RGBA"):
    color_size = 4
else: #color_mode is RGB
    color_size = 3
input_shape = (config["exec"]["define_model"]["params"]["input_size"][0],config["exec"]["define_model"]["params"]["input_size"][1],color_size)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # converting 2D array into fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(rate=config["exec"]["define_model"]["params"]["dropout_rate"], seed=12345))
model.add(Dense(config["exec"]["define_model"]["params"]["nb_classes"],activation=tf.nn.softmax))

model = compile_model(model=model,
                      optimizer=get_optimizer(**config["exec"]["get_optimizer"]["params"]),
                      loss=get_loss_function(**config["exec"]["get_loss_function"]["params"]),
                      **config["exec"]["compile_model"]["params"])


# ### Fit the model

# Callbacks for monitoring training process

# In[ ]:


callbacks = []
for callback in config["init"]["callbacks"]:
    callbacks = callbacks + [get_and_log(keras_containers.CallBacks, callback["name"])(callback["params"])]


# In[ ]:


model = fit(model = model,
            data_bunch = data_bunch_fit,
            callbacks = callbacks,
            **config["exec"]["fit"]["params"])


# #### Save the model

# In[ ]:


save_model_dict = {
    save_model_function_name: get_and_log(keras_containers.ModelSavers, save_model_function_name) for save_model_function_name in config["init"]["save_model"]["names"]
}
for model_format, save_model in save_model_dict.items():
    
    tasks.store_model(save_model=save_model,
                      model=model,
                      copy_from_local_to_remote = get_and_log(common_containers.ArtifactCopiers, config["init"]["copy_from_local_to_remote"]["name"]),
                      **config["exec"]["save_model"][model_format]
                      )


# #### Evaluate model

# In[ ]:


evaluate = get_and_log(keras_containers.ModelEvaluators, config["init"]["evaluate"]["name"])
predict = get_and_log(keras_containers.PredictionFunctions, config["init"]["predict"]["name"])
custom_label_metrics_dict = {
    custom_label_metric_name: get_and_log(common_containers.CustomLabelMetrics, custom_label_metric_name) for custom_label_metric_name in config["init"]["custom_label_metrics"]["names"]
}


# In[ ]:


data_bunch_metrics = maybe_transform(data_bunch_fit, config["exec"]["evaluate"].get("pre_execution_transformation"))
data_bunch_predict = maybe_transform(data_bunch_metrics, config["exec"]["predict"].get("pre_execution_transformation"))

for data_set_name in config["exec"]["predict"]["data_set_names"]:
    data_set = getattr(data_bunch_predict, data_set_name)
    data_set.predictions = predict(model=model, data_set=data_set, **config["exec"]["predict"]["params"])


# In[ ]:


data_bunch_metrics = maybe_transform(data_bunch_fit, config["exec"]["evaluate"].get("pre_execution_transformation"))
metrics = {}
for data_set_name in config["exec"]["evaluate"]["data_set_names"]:
    data_set = getattr(data_bunch_metrics, data_set_name)
    metrics[data_set_name] = evaluate(model, data_set, **config["exec"]["evaluate"]["params"])
print(json.dumps(metrics, indent=2))


# In[ ]:



data_bunch_custom_metrics = maybe_transform(data_bunch_predict, 
                                            config["exec"]["evaluate_custom_metrics"].get("pre_execution_transformation"))
custom_label_metrics = {}
for data_set_name in config["exec"]["evaluate_custom_label_metrics"]["data_set_names"]:
    data_set = getattr(data_bunch_custom_metrics, data_set_name)
    custom_label_metrics[data_set_name] = tasks.evaluate_label_metrics(data_set, custom_label_metrics_dict)
print(json.dumps(custom_label_metrics, indent=2))

