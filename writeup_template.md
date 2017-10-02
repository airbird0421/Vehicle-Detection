##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image0]: ./examples/vehicle.png
[image1]: ./examples/non-vehicle.png
[image20]: ./examples/car-ch0-hog.png
[image21]: ./examples/not-car-ch0-hog.png
[image3]: ./examples/sliding_window.jpg
[image4]: ./examples/sliding_window_example.png
[image5]: ./examples/heat-map.png
[image6]: ./examples/label.png
[image7]: ./examples/boxes.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2nd code cell of the IPython notebook, and the function name is `get_hog_features`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![vehicle][image0]
![non-vehicle][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example (corresponding to the two images show above) using the `YCrCb` color space channel 0, and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![vehicle-hog][image20]
![non-vehicle-hog][image21]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Basically, when I try to tune one parameter, I fix other parameters, and run several times on the same dataset and compare the accuracy. With this method, the final parameters which I believe best are: orientations:12, pixels_per_cell:8, cell_per_block:1 or 2 (but 1 saves a lot of training time, so I used 1 in the end). Also, I found using hog features on all 3 channels are better than on any single channel.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using all three kinds of features. How the HOG features are extracted from an image is already described above. For raw color features, and color histogram features, I used similar methods, i.e. when tuning one parameter, fix others. Based on same dataset, by comparing the accuracy, for color histogram features, the final bin number working best for me is 128; the final size for raw-pixel features that can achieve good enough results is 16x16. Also, I tried different color spaces with each of those three kinds of features, and the final selection is YCrCb, which seems best for the final result.

After I extract features from all the images, I used `StandardScaler()` from the `sklearn.preprocessing` module to normalize the features, so that they have 0 mean and same range of variance. I used `train_test_split` from `sklearn` to split the dataset into training set and test set, and used training set to fit the linear SVC, and test set to measure it's performance. The final accuracy basically is stable, higher than 0.99.

The raw color feature and color histogram feature extraction is in the 2nd code cell in IPython notebook, with function name `bin_spatial` and `color_hist` respectively. The linear SVM model training is also in the 4th code cell.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For the sliding window search, first thing is to have a function `find_cars()` in the 5th code cell, which takes an area to search, the scaling factor, and searches in all the sliding windows in that area. When doing the search, need to extract HOG, raw color, and color histogram features from each window and then predict using the trained linear model. The HOG feature can be got only once for the whole searching area and no need to run HOG for each window to save processing time.

Second thing is to decide which erea to search and what scale factor to use in each area. Basically, I tried to check the size of car in different areas of the image. The closer, the bigger it is, and the bigger scale factor to use. When it goes farther, I only need to search that area with a smaller window, or smaller scale factor. The more windows to search, or the bigger the overlap is, the more chance to find a car, but also more time for processing. This is a trade off. In the end, I selected 4 kinds of areas, with scale factors of 1, 1.5, 2, and 3, repectively, and an overlap rate of 0.75. The corresponding code is in the `pipeline` function in the 5th code cell.

![all sliding windows][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Basically, I tried to adjust the search areas as stated above, and removed unnecessary scales, like 4 to optimize the pipeline. The performance of the classifier is already fixed once the mode is trained. I set `cells_per_block` to 1, which I believe can greatly reduce the overall feature size and save processing time. Ultimately I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![pipeline results][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. The thresholding can eliminate some of the false positives.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![frames and their heatmap][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![label][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![detection result for the last frame][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

First, it's about the tuning of number of frames to do heatmap integration and the threshoding. Since on one image, the detection may not be that stable, sometimes it may not detect the car, sometimes it may give false positives. The obvious solution is to do some kinds of "smoothing", i.e. take advantage of the results of several consecutive images since the cars' position on several consecutive frames are almost the same. When integrating several frames together, on frames where no cars detected, if the previous several frames have cars detected, then it can still show the detection; on frames where there'are random false positives which, in most cases, don't happen continuously, those can be eliminated by means of proper threshoding. But the number of thresholding is also a tradeoff. If the number is too high, false positives can be removed effectively, but real car detections can also be removed; if the number is too low, there will be more false detections. My final result is 10 over 10 frames, which gives a decent result.

Second, the false positives. At the beginning, When I used HLS color space, I did see some false positives in a certain areas, like shadow areas, on the guardrails. Even after I tuned the threasholding carefully, I still couldn't completely eliminate them. I even went back to change the features to use, like HOG only, changing parameters, but didn't get obvious improvements. But later, I tried to use different color spaces, and found that YCrCb gave a realy good result. It can effectively remove almost all false positives in shadows and also on the guardrails. But YCrCb did redudced the car detections as well. I had to adjust the thresholding again. But the final result is quite satifying when I used threshold 10 over 10 frames.

Third, the wobbly image. In the video, the bounding box is wobbling all the time, on both the size and the position. I think this comes from the slinding windows. We used different sized windows, and the window slides in a non-continous fashion. So on different frames, the bounding boxes may appear in different positions, and with different sizes. One way I can think of to handle it is to apply some kinds of calculation, i.e. if the bounding box in one frame overlaps with that in the previous frame, and the overlapping exceeds some threshold, then we can think it's for the same car, in this case, we don't allow it to change all the way to its current size and/or position, rather, only allow it to change a little which may seems better to human eyes.



