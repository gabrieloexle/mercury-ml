
# coding: utf-8

# ## Imports

# In[ ]:


import sys
import os
sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0])
import uuid
import json
import copy

from mercury_ml.common import tasks
from mercury_ml.common import utils
from mercury_ml.common import containers as common_containers
from mercury_ml.keras import containers as keras_containers


# In[ ]:


session_id = str(uuid.uuid4().hex)


# ### load config file and update placeholder

# In[ ]:


data_bunch_name= "images_456"


# ## Source
# First set the parameter for the source

# In[ ]:


read_source_train_param = {
          "generator_params": {
            "channel_shift_range": 0.0,
            "data_format": "channels_last",
            "featurewise_center": True,
            "featurewise_std_normalization": True,
            "fill_mode": "nearest",
            "height_shift_range": 0.1,
            "horizontal_flip": True,
            "rescale": 0.00392156862745098,
            "rotation_range": 0.2,
            "samplewise_center": True,
            "samplewise_std_normalization": True,
            "shear_range": 0.1,
            "vertical_flip": True,
            "width_shift_range": 0.1,
            "zca_epsilon": 1e-6,
            "zca_whitening": False,
            "zoom_range": 0.1
          },
          "iterator_params": {
            "directory": "./example_data/"+data_bunch_name+"/train",
            "batch_size": 2,
            "class_mode": "categorical",
            "color_mode": "rgb",
            "seed": 12345,
            "shuffle": True,
            "target_size": [
              10,
              10
            ]
          }
        }

read_source_data_parm_valid = copy.deepcopy(read_source_train_param)
read_source_dara_parm_test =  copy.deepcopy(read_source_train_param)
read_source_data_parm_valid["iterator_params"]["shuffle"]= False
read_source_dara_parm_test["iterator_params"]["shuffle"]= False

read_source_data_parm_valid["iterator_params"]["directory"]= "./example_data/"+data_bunch_name+"/valid"
read_source_dara_parm_test["iterator_params"]["directory"]= "./example_data/"+data_bunch_name+"/test"

read_source_data_set = common_containers.SourceReaders.read_disk_keras_single_input_iterator
data_bunch_fit = tasks.read_train_valid_test_data_bunch(read_source_data_set,
                                                        read_source_train_param,
                                                        read_source_data_parm_valid,
                                                        read_source_dara_parm_test)

####################################################
# feature normalization
# PART 1
mean = None
std = None
pca = None
if not {"featurewise_center", "featurewise_std_normalization", "zca_whitening"}.isdisjoint(read_source_train_param["generator_params"].keys()):

    import numpy as np
    from keras.preprocessing import image

    #files = data_bunch_fit.train.targets.underlying.image_data_generator
    files = data_bunch_fit.train.targets.underlying.filenames

    images = []
    for file in files:
        img = image.load_img(
            os.path.join(read_source_train_param["iterator_params"]["directory"], file),
            color_mode=read_source_train_param["iterator_params"]["color_mode"],
        target_size=(10,10))
        img_array = image.img_to_array(img)
        images.append(img_array)

    data_bunch_fit.train.targets.underlying.image_data_generator.fit(images)

    mean = data_bunch_fit.train.targets.underlying.image_data_generator.mean
    std = data_bunch_fit.train.targets.underlying.image_data_generator.std
    pca = data_bunch_fit.train.targets.underlying.image_data_generator.principal_components

    # todo investigation needed if copying the parameter is enough
    # alternatively, maybe its better to call
    # data_bunch_fit.test.targets.underlying.image_data_generator.fit(images)
    # data_bunch_fit.valid.targets.underlying.image_data_generator.fit(images)

    data_bunch_fit.test.targets.underlying.image_data_generator.mean = mean
    data_bunch_fit.test.targets.underlying.image_data_generator.std = std
    data_bunch_fit.test.targets.underlying.image_data_generator.principal_components = pca

    data_bunch_fit.valid.targets.underlying.image_data_generator.mean = mean
    data_bunch_fit.valid.targets.underlying.image_data_generator.std = std
    data_bunch_fit.valid.targets.underlying.image_data_generator.principal_components = pca

# END
# feature normalization PART 1
#########################################################

optimizer_parm = {
        "optimizer_name": "adam",
        "optimizer_params": {
          "lr": 0.001
        }
      }
optimizer =keras_containers.OptimizerFetchers.get_keras_optimizer(**optimizer_parm)
loss_function = keras_containers.LossFunctionFetchers.get_keras_loss("categorical_crossentropy")




model = keras_containers.ModelDefinitions.define_conv_simple(
    input_size = [10, 10],
    nb_classes=2,
    final_activation="softmax",
    dropout_rate=0.1)


# compile model

# In[ ]:


model = keras_containers.ModelCompilers.compile_model(model=model,
                      optimizer=optimizer,
                      loss=loss_function,
                      metrics=["acc"])


# ### Fit the model

# Callbacks for monitoring training process

# In[ ]:


callback_params_early_st = {
          "patience": 2,
          "monitor": "val_loss",
          "min_delta": 0.001
        }
callback_params_early_model_ch = {
    "filepath": "./example_results/local/"+session_id+"/model_checkpoint/last_best_model.h5",
    "save_best_only": True
}
callbacks = [keras_containers.CallBacks.early_stopping(callback_params_early_st),
             keras_containers.CallBacks.model_checkpoint(callback_params_early_model_ch)]


# Train the model

# In[ ]:


model = keras_containers.ModelFitters.fit_generator(model = model,
            data_bunch = data_bunch_fit,
            callbacks = callbacks,
            epochs=5)


# #### Save the model

# specify model savers

# In[ ]:


save_model_dict = {"save_hdf5":keras_containers.ModelSavers.save_hdf5,
"save_tensorflow_serving_predict_signature_def":keras_containers.ModelSavers.save_tensorflow_serving_predict_signature_def}


model_local_dir ="./example_results/local/"+session_id+"/models"
model_remote_dir = "./example_results/remote/"+session_id+"/models"
model_object_name= "fit_example__"+session_id
save_model_parm = {
      "save_hdf5": {
        "local_dir": model_local_dir,
        "remote_dir": model_remote_dir,
        "filename": model_object_name+"__hdf5",
        "extension": ".h5",
        "overwrite_remote": True
      },
      "save_tensorflow_serving_predict_signature_def": {
        "local_dir": model_local_dir,
        "remote_dir": model_remote_dir,
        "filename": model_object_name+"__tf_serving_predict",
        "temp_base_dir": "c:/tf_serving/_tmp_model/"+model_object_name+"__tf_serving_predict",
        "extension": ".zip",
        "overwrite_remote": True,
        "do_save_labels_txt": True,
        "input_name": "input",
        "output_name": "output",
        "labels_list": ["cat","dog"]
      }
    }


# save model

# In[ ]:


for model_format, save_model in save_model_dict.items():
    
    tasks.store_model(save_model=save_model,
                      model=model,
                      copy_from_local_to_remote = common_containers.ArtifactCopiers.copy_from_disk_to_disk,#get_and_log(common_containers.ArtifactCopiers, config["init"]["copy_from_local_to_remote"]["name"]),
                      **save_model_parm[model_format]
                      )

## load Model
load_model_param = {
      "local_dir": save_model_parm["save_hdf5"]["local_dir"],#[]
      "remote_dir": save_model_parm["save_hdf5"]["remote_dir"],
      "filename": save_model_parm["save_hdf5"]["filename"],
      "extension": save_model_parm["save_hdf5"]["extension"],
      "always_fetch_remote": False
    }
loaded_model = tasks.load_model(load_model=keras_containers.ModelLoaders.load_hdf5,
                         copy_from_remote_to_local=common_containers.ArtifactCopiers.copy_from_disk_to_disk,
                         custom_objects = {"categorical_crossentropy":loss_function},
                         **load_model_param
                        )

### load test data

data_bunch_test = tasks.read_test_data_bunch(read_source_data_set,read_source_dara_parm_test)


##############################################
# feature normalization
# PART 2
data_bunch_test.test.targets.underlying.image_data_generator.mean = mean
data_bunch_test.test.targets.underlying.image_data_generator.std = std
data_bunch_test.test.targets.underlying.image_data_generator.principal_components = pca
# END
# feature normalization PART 2
##############################################

evaluate = keras_containers.ModelEvaluators.evaluate_generator
predict = keras_containers.PredictionFunctions.predict_generator

custom_label_metrics_dict = {"evaluate_numpy_accuracy":common_containers.CustomLabelMetrics.evaluate_numpy_accuracy,
        "evaluate_numpy_confusion_matrix":common_containers.CustomLabelMetrics.evaluate_numpy_confusion_matrix}


# In[ ]:


data_bunch_test.test.predictions = predict(model=loaded_model, data_set=data_bunch_fit.test)


# In[ ]:


result = evaluate(loaded_model, data_bunch_fit.test)
print(json.dumps(result, indent=2))


# transform test images with numpy

# In[ ]:


transformation_param = {
        "data_set_names": ["test"],
        "params": {
          "transform_to": "numpy",
          "data_wrapper_params": {
            "predictions": {},
            "index": {},
            "targets": {}
          }
        }
      }
data_bunch_test_predict = data_bunch_test.transform(**transformation_param)

# In[ ]:


confMat = tasks.evaluate_label_metrics(data_bunch_test_predict.test, custom_label_metrics_dict)
print(json.dumps(confMat, indent=2))

