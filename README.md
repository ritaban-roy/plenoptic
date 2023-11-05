## Instructions

There are two files in the `src` directory
```
plenoptic.py -> contains the code for question 1 and 3

extract_frames.py -> Extracts frames from the video for question 3
```

In `plenoptic.py`, here are explainations for the important functions for question 1

1. `load_lightfield(img_path)` : loads the plenoptic image and returns a 5D lightfield 
2. `create_mosaic()` : creates the mosaic for the plenoptic image
3. `create_focal_stack()` : creates a focal stack and and saves it a `.npz` file
4. `refocus(lightfield, depth)` : takes a lightfield and a depth value and creates the image that is focused at `depth`
5. `depth_from_focus()` : creates an all in focus image and depth map from a focal stack
6. `refocus_aperture(lightfield, aperture, depths)` : takes a lightfield, an aperture and a set of depth and return the focal stack at that `aperture`
7. `create_confocal_stack()` : creates a focus-aperture stack by calling `refocus_aperture` with different apertures
8. `confocal_stereo()` : save a depth map estimated from the AFIs

To get the results, call `run_q1_focal()` , to get the all in focus image and the depth map, along with the images focused at different depths

Call `run_q1_confocal()` to get the depth map from the focus aperture stack

Call `run_q3()` to get the results for question 3. It will first prompt you to select a point, which will be used to select the template, and then save the image which is focused at that depth where the point belongs.

Run `extract_frames.py` using
```
python extract_frames.py
```
to get all the necessary frames from the video. This must be run before calling `run_q3()` in `plenoptic.py`

NOTE : If your process gets killed at any stage due to insufficient memory or otherwise, the functions called from `run_q1_focal()`, `run_q1_confocal()` and `run_q3()` can be run individually in the same order. Every function saves their result in .npz files which can be used for the subsequent steps