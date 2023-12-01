# 3D Histology Reconstruction

This repository hosts the code for 3D histology reconstruction using a probabilistic model of spatial deformation that yields 
reconstructions for multiple histological stains that are jointly smooth, robust to outliers and follow the reference shape.

The framework is used for accurate 3D reconstruction of two stains (Nissl and parvalbumin) from the Allen human brain atlas (/scripts). Two different base registration algorithms are used: learning-based (RegNet) using PyTorch and a more standard approach using NiftyReg package.

### Requirements:
**Python** <br />
The code run on python v3.6.9 and several external libraries listed under requirements.txt

**Gurobi package** <br />
Gurobi optimizer is used to solve the linear program. [You can download it from here](https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_using_pip_to_install_gr.html) and [create an academic free license from here](https://www.gurobi.com/documentation/9.1/quickstart_mac/creating_a_new_academic_li.html#subsection:createacademiclicense)

**NiftyReg package** <br />
Needed to run the algorithm using NiftyReg as base registration algorithm. 
http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_documentation

**MATLAB** <br />
Needed to download data from the Allen repository.

### Run the code
- **Set-up configuration files** 
  - _setup.py_: original absolute paths are specified here. Please, update this according to your setting.
  - _config_dev.py_: experiment parameters are specified here, specially regarding neural network training (loss, network, etc...)
  - _config_data.py_: data parameters are specified here. You may not want to modify this file

- **Download data from the Allen repository** <br />
  From /database/preprocessing, run the _download_allen.m_ script to download all sections and generate the corresponding masks. The initial linear alignment between the histology stack and the MR volume would be also downloaded from our public data repository (XXXX). Finally, you'll also find the MNI template (T2 contrast) and the correspondig segmentation for mapping the reconstructions to the MNI space.
  
- **Create dataset** <br />
  From /database/preprocessing, run the _generate_initial_images.py_ file in order to set the database properly to be used in the code (naming convention, ordering, etc...). The _slice\_info\_\* .csv_ files are used to map images from the Allen repository to the database convention used in this project.
  
 
- **Generate images** <br />
  From /scripts, run the _generate_slices.py_ file to generate the images that will be used in the project, at the appropriate resolution and shape.

- **Train the registration networks** <br />
  Under /networks, there are several script to train and test intra and intermodality registration networks. You can run any of these from the command line with the appropriate parameters and configuration (_config_dev.py_)

- **Build the graph and run the algorithm** <br />
  Under /algorithm there are the scripts to build the graph ( _initialize_graph_X.py_ ) and run the algorithm (_algorithm_X.py_ ) for RegNet and NR registration algorithms. 

- **Group blocks** <br />
  Run the _group_blocks.py_ file under /scripts/visualization. This wil group the _*.tree_ images in a single file. You need to specify which reconstruction you want to group.

- **Deform colored images** <br />
  Use the _deform_images.py_ file under /scripts/visualization to deform the original colored images from each contrast and create a _\*\.tree_ file for each.

- **Equalize and resample** <br />
  Use the _eq_and_resample.py_ file under /scripts/visualization for simple equalization of histology sections and resampling along the stack directon at the desired resolution.



## Code updates

26 March 2021:
- Initial commit

## Citation
Casamitjana, Adrià, et al. "Robust joint registration of multiple stains and MRI for multimodal 3D histology reconstruction: Application to the Allen human brain atlas." Medical image analysis 75 (2022): 102265.

## References
[1] [Model-based refinement of nonlinear registrations in 3D histology reconstruction](https://www.nmr.mgh.harvard.edu/~iglesias/pdf/MICCAI_2018_histoRecon.pdf)
Juan Eugenio Iglesias, Marco Lorenzi, Sebastiano Ferraris, Loïc Peter, Marc Modat, Allison Stevens, Bruce Fischl, and Tom Vercauteren
MICCAI 2018

[2] [VoxelMorph: A Learning Framework for Deformable Medical Image Registration](https://arxiv.org/abs/1809.05231)
Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
TMI 2019
