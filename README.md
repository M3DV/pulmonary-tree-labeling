# pulmonary-tree-labeling
[[MedIA](https://doi.org/10.1016/j.media.2024.103367)] Efficient anatomical labeling of pulmonary tree structures via deep point-graph representation-based implicit fields 

Preprint: [arXiv:2309](https://arxiv.org/abs/2309.17329)


## :bookmark_tabs:Data Preparation

We utilized the Pulmonary Tree Labeling dataset ([PubMed](https://pubmed.ncbi.nlm.nih.gov/21452728/#:~:text=Methods)), which includes 799 subjects, including the 3D volumes of binary shapes of pulmonary airway, artery and vein for each subject. The dataset is divided into 559 subjects for training, 80 cases for validation, and 160 cases for testing. 


Furthermore, we provide pre-generated images with lesions based on the `LIDC-IDRI\Normal` dataset. These images are stored in `LIDC-IDRI\Demo`, where `Image_i` represents the images generated under the control information *hist_i*. The pre-trained weights used to generate these images are available in the pre-trained weights mentioned below ([HuggingFace🤗](https://huggingface.co/YuheLiuu/LeFusion/tree/main/LIDC_LeFusion_Model)).

## :Specifications

We have included a folder 'specs' that contains detailed specfication for this project.

'network_specs.json' --- all configuration of the implicit point graph network.
    ["Graph_Encoder"]
        ["Network_Name"]: Name of the graph-based initial encoder network.
        ["GNN_Hidden_Layers"]: the feature dimension of each graph layer as a list.
        ["Attention_Head_Num"]: The number of attention head in all the attention layers in GAT.
        ["Input_Channel"]: The number of dimension representing each node as input.
    ["Point_Encoder"]
        ["Network_Name"]: Name of the point-based initial encoder network.
        ["PointAbstractions"]: The number of points to be abstracted at each pointnet layer. 
        ["Out_Channel_Dim"]: The feature dimension that the model output.
    ["Point_Graph_Fusion"]
        ["initial_feature_dim"]: Initial feature dimension from the initial point and graph encoders.
        ["p_operation"]: The operation selection for the point cloud to acquire graph-based context. 1 = feature propagation, 0 = ball query
        ["n_operation"]: The operation selection for the graph nodes to acquire point-based context. 1 = feature propagation, 0 = ball query
        ["blks_dim"]: The dimension of feature output from each point-graph fusion layer.
        ["fp_nums"]: The number of closest element from the opposite branch to search for and propagate feature.
        ["radius"]: The radius of each point-graph fusion layer in the ball query operation.
        ["nsample"]: The max number of samples from the ball query search result that will be used.
    ["Implicit_Module"]
        ["pgn_layer_imp_input"]: The indices of the layers whose feature output will be used for the final implicit point module training and inference.

'train_inference_specs' --- training, inference related parameters.
Common configurations:
    ["Load_Model"]: 1 - load pretrained model; 0 - don't load pretrained model;
    ["Save_Model"]: 1 - save model; 0 - don't save model;
    ["Batch_Size"]: batch size for training and validation.
    ["Learning_Rate"]: The initial learning rate during training.
    ["Num_Epoches"]: The number of training epoches.
    ["Lr_Decay_Epoches"]: The number of epoches for a learning rate decay.
    ["Learning_Rate_Decay"]: The ratio between the decayed and the original learning rate. 
    ["Validation_Per_Epoches"]: Perform validation after the number of training epoches.
    ["Snapshot_Frequency"]: The frequency of epoches to print the current training statistics.
    ["checkpoint_filename"]: The file name of the checkpoint.
    ["Load_Model"]: 1 - load pretrained model; 0 - don't load pretrained model;
    ["Save_Model"]: 1 - save model; 0 - don't save model;
    ["Implicit_Network_inference_Specs"]["Make_Report"]: If output a performance report of the reconstructed volumes.
    ["Report_file_Name"]: file name of the report.

'dataset_specs_XXX' --- dataset specific details
Common configurations:
    ["Dataset_Name"]: The pulmonary component.
    ["Max_Node"]: The max number of node a graph in this dataset could have.(required to be pre-determined)
    ["Num_Class"]: Number of class for the segmentation task.
    ["Model_Save_Path"]: Model's checkpoint directory(relative path).
    ["Training_Dir"]: The number of training epoches.
    ["Validation_Dir"]: The number of epoches for a learning rate decay.
    ["Testing_Dir"]: The ratio between the decayed and the original learning rate. 
    ["Inference_Dir"]: Perform validation after the number of training epoches.
    ["Result_Save_Path"]: The frequency of epoches to print the current training statistics.
    ["Save_Reconstruction_Volume"]: The file name of the checkpoint.
    ["Make_Report"]: 1 - load pretrained model; 0 - don't load pretrained model;
    ["Save_Model"]: 1 - save model; 0 - don't save model;
    ["Report_file_Name"]: file name of the report.

'dataset_specs_XXX.json' --- details regarding the dataset that the program will run on. 

## :nut_and_bolt: Installation

1. Create a virtual environment `conda create -n ipgn python=3.9` and activate it `conda activate ipgn`
2. Download the code`git clone https://github.com/M3DV/pulmonary-tree-labeling.git`
3. Enter the project folder `cd pulmonary-tree-labeling` and run `conda install --yes --file requirements.txt`

## :bulb:Get Started

1. Download the PTL datasets (airway, artery, vein) ([Google Drive](https://drive.google.com/drive/folders/1Fi088yjdRgmXbI629hZuXoQfHMbF2gcV?usp=sharing))

   optional: Download the derived Graph skeleton (airway, artery, vein) dataset ([Google Drive](https://drive.google.com/drive/folders/1Fi088yjdRgmXbI629hZuXoQfHMbF2gcV?usp=sharing))

Place the downloaded PTL datasets to under the `pulmonary-tree-labeling` directory.

2. Download the pre-trained Model Checkpoints([Google Drive](https://drive.google.com/drive/folders/1ursbfZQY0D9cFoLenGZUqjP8TUQ03UxE?usp=sharing))

   We offer the pre-trained IPGN models, which has been trained on the PTL dataset independently on airway, artery and vein. This pre-trained model can be directly used for Inference if you do not want to re-train the Model. Simply download the folder to under the `pulmonary-tree-labeling` directory.

3. Decide the dataset that the algorithm will be running on:
   The specification file for each pulmonary structure dataset are in the 'specs' folder, and with json filename starting with 'dataset_specs'. 
   e.g. dataset_specs_airway.json

## :microscope:Train IPGN

Training from stretch:

> ✨**Note**: Before running the following command, make sure you are inside the 'pulmonary-tree-labeling' folder.

```bash
python train_network.py -d $dataset_specs_json_file 
```
e.g. python train_network.py -d dataset_specs_airway.json


## :chart_with_upwards_trend:Inference

Start inference:

> ✨**Note**: Before running the following command, make sure you are inside the `LeFusion/LeFusion_LIDC` folder.

```bash
python reconstruction_inference.py -d $dataset_specs_json_file 
```

In the dataset specs file, if "Save_Reconstruction_Volume" is set to 1, the script will save the reconstructed 3D volume to the "Result_Save_Path"
as directory.

## Citation

```
@article{XIE2025103367,
title = {Efficient anatomical labeling of pulmonary tree structures via deep point-graph representation-based implicit fields},
journal = {Medical Image Analysis},
volume = {99},
pages = {103367},
year = {2025},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2024.103367},
url = {https://www.sciencedirect.com/science/article/pii/S1361841524002925},
author = {Kangxian Xie and Jiancheng Yang and Donglai Wei and Ziqiao Weng and Pascal Fua},
keywords = {Pulmonary tree labeling, Graph, Point cloud, Implicit function, 3D deep learning}
}
```

## Acknowledgement

Some of our code is modified based on [Pointnet++](https://github.com/charlesq34/pointnet2) and we greatly appreciate the efforts of the respective authors for providing open-source code. 


