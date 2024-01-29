 # BOA::Contrast

Package to compute contrast information from a CT image, part of the [BOA](https://github.com/UMEssen/Body-and-Organ-Analyzer). The package uses the open-source software [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) to compute segmentations of important anatomical landmarks, which are then used to create features for a machine learning model to predict the contrast information.

## Install

```bash
pip install boa-contrast
```

will install only the basic package (without the TotalSegmentator), if you also want to install the TotalSegmentator you can

```bash
pip install "boa-contrast[totalsegmentator]"
```

However, the TotalSegmentator can also be used together with docker, and in such case it is not needed to install it.

## Command Line
```
constrast-recognition --help
```
Once a CT and a folder where to store the TotalSegmentator segmentations is given, you can run it using the following command
```
contrast-recognition [-h] --ct-path CT_PATH --segmentation-folder SEGMENTATION_FOLDER [--docker] [--user-id USER_ID] [--device-id DEVICE_ID] [-v]
```

You can run it using docker by using the `--docker` flag. If you are using docker, you need to specify your user ID using the `--user-id` flag, otherwise you will have to change the ownership of the segmentations afterwards.

If you are using a GPU, you can specify the device ID using the `--device-id` flag.

You can enable verbosity with the `-v` flag.

To not download the TotalSegmentator weights all the time, you can specify their location using the `TOTALSEG_WEIGHTS_PATH` environment variable.

A sample output looks as follows:
```
IV Phase: NON_CONTRAST
Contrast in GIT: NO_CONTRAST_IN_GI_TRACT
```

## From Python
Compute the segmentation with the TotalSegmentator with docker

```python
from boa_contrast import compute_segmentation

compute_segmentation(
    ct_path=...,  # The path to the CT
    segmentation_folder=...,  # The root where the segmentation should be stored
    device_id=...,  # The ID of the GPU device or -1
    user_id=...,  # Your user ID for docker to run in user mode
    compute_with_docker=False,  # Whether to use docker or not
)
```

Once the segmentation is computed

```python
from boa_contrast import predict

predict(
    ct_path=...,  # path to the CT
    segmentation_folder=...,  # path to this CT's segmentation
)
```

Output:
```
{
    "phase_ensemble_prediction": 0,
    "phase_ensemble_predicted_class": "NON_CONTRAST",
    "phase_ensemble_probas": array(
        [
            9.89733540e-01,
            3.60637282e-04,
            4.79974664e-04,
            5.55973168e-04,
            8.86987492e-03,
        ]
    ),
    "git_ensemble_prediction": 0,
    "git_ensemble_predicted_class": "NO_CONTRAST_IN_GI_TRACT",
    "git_ensemble_probas": array(
        [
            9.99951577e-01,
            4.84187825e-05,
        ]
    ),
}
```
