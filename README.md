# Medical image segmentation in Tensorflow

Here, you will find Tensorflow based code for segmenting medical images. Make sure to have python packages such as h5py, SimpleITK and TensorFlow installed.

Assume that you have Niftii files (.nii.gz) and all the training images are stored in a folder with training subjects as different folders. Each training subject will have a folder name that matches the name of the CT image. The groundtruth image should be in the same folder, named as GT.nii.gz. Each voxel in the groundtruth image should have values ranging from 0 to num_classes-1. See the structure below: 

```
Data
|
|--sub1/
    |--sub1.nii.gz
    |--GT.nii.gz
|--sub2/
    |--sub2.nii.gz
    |--GT.nii.gz
|
|...
```

CT file and folder name can be different as long as they match.

Initially, you would have to convert the CT and groundtruth data to h5 format using the generate_2d_h5.py script. To run it, type:

python generate_2d_h5.py --src /path/to/patients --dst /path/to/save/h5/files


This generates a folder with several h5 files that contain the training data (input ct slices and their corresponding labels).

Afterwards, you will be able to run:

python main.py --dir_patients /path/to/CT_data --path_patients_h5 /path/to/h5files

The codes come with several options for training and testing that you'll see when you run python main.py
