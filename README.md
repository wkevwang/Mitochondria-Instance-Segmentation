# Instance Segmentation of Mitochondria

Experiments with deep learning for counting and quantifying the area of mitochondria in images

## Problem Background
In biology research, it is a common time-consuming task to manually label objects in microscope images. For example, to study the correlation between mitochondria count/size and heart disease, a researcher must quantify mitochondria in hundreds to thousands of images (this is the real project my colleagues are working on). The labelling may involve manually drawing polygons around instances of mitochondria, as shown below. The mitochondria are the elliptical objects with encapsulated lines and circles.

![labelled mitochondria image](https://github.com/Megasonic988/Mitochondria-Instance-Segmentation/blob/master/Sample_Data/ImageJ/Image_1.png)

A system that can automate visual identification tasks such as mitochondria labelling would save researchers countless hours in their work. Automating the labelling of mitochondria is a perfect task for deep learning, especially through the usage of semantic segmentation/instance segmentation techniques. Through this project, I aim to learn about deep learning methods for solving computer vision problems while applying these methods to solve a real world task.

## Deep Learning Methods
To solve this problem, I aim to teach the program to perform instance segmentation on the mitochondria (i.e. to color-fill each mitochondria). This approach will allow us to use the incredible research on convolutional neural networks and instance segmentation to solve this problem. After segmenting, we can easily extract the count and area of mitochondria.

#### My plan:
1. Train a Fully Convolutional Network (FCN) to perform semantic segmentation on mitochondria (coloring all mitochondria as a single class, without regard to individual instances). Starting with FCN is simpler and will be a good gauge of the promise of deep learning methods
2. Train Mask R-CNN to perform instance segmentation on mitochondria. Mask R-CNN combines a bounding box identifier for individual instances of objects with a segmentation model, allowing the color-filling of individual instances of objects. I choose Mask RCNN because it is currently the state-of-the-art model for instance segmentation, with [amazing results](https://www.youtube.com/watch?v=OOT3UIXZztE).
The results of training the FCN were promising; here, I focus on the results process and results of training Mask R-CNN.

## Dataset
The labelled dataset comes from a biology research lab at my university. It consists of:
* 50 images at 4864x3704 resolution
* 1036 instances of mitochondria

The images are at a very high resolution, containing much fine-grained detail that is not necessary for understanding mitochondria structure. To avoid saturating the network with learning this low level detail, I resized the images 0.25x. Furthermore, since there are a limited number of images, I applied data augmentation techniques to maximize the amount of training data I had, including horizontal mirroring and a 180 degree rotation to obtain 4x the original amount of data.

A sample of the training data can be found in [Sample_Data](https://github.com/Megasonic988/Mitochondria-Instance-Segmentation/tree/master/Sample_Data).

## Training
I use [Matterport's Mask RCNN](https://github.com/matterport/Mask_RCNN) implementation. This model is already pretrained on the MS COCO dataset, so my training is fine-tuning the model.

#### Details of the training:
* 2 classes (mitochondria and background)
* 80/20 train/validation split (high train proportion because I have a limited dataset)
* 20 epochs (empirically the validation loss did not decrease significantly after 20 epochs)
* learning rate of 1e-4 (empirically found that the lowest validation loss is achieved with this value)
* training on all layers
* other Mask R-CNN parameters left at defaults of the Matterport implementation

I trained on a Nvidia P100 GPU on Google Cloud Platform. The training takes approximately 40 minutes.

## Results
The resulting model labels with astoundingly high success:
![Labelled image](https://github.com/Megasonic988/Mitochondria-Instance-Segmentation/blob/master/Sample_Results/Image_1_Output.png)
![Labelled image](https://github.com/Megasonic988/Mitochondria-Instance-Segmentation/blob/master/Sample_Results/Image_2_Output.png)

A comparison of the prediction (right) with ground truth (left):
![Comparison](https://github.com/Megasonic988/Mitochondria-Instance-Segmentation/blob/master/Sample_Results/Image_1_Comparison.png)
![Comparison](https://github.com/Megasonic988/Mitochondria-Instance-Segmentation/blob/master/Sample_Results/Image_2_Comparison.png)

As one can see, the model has trouble with
* labelling one mitochondria as two mitochondria. The model thinks the internal lines within the mitochondria are actually the walls between two separate mitochondria
* labelling two mitochondria as one mitochondria
* identification of other elliptical objects as mitochondria
* ignoring mitochondria

More test images are available in [Sample Results](https://github.com/Megasonic988/Mitochondria-Instance-Segmentation/tree/master/Sample_Results).

#### Discussion
For such a limited number of images, the results are quite remarkable. I believe the success results from these factors:
1. Fine-tuning on a pre-trained network. Because the Mask R-CNN implementation I used was already trained on MS COCO, the feature extractors of the model already could identify complex high-level features. Because mitochondria are not too visually complex, it is easy to learn how to identify mitochondria with fine-tuning.
2. Two-class simplicity of problem. Because there were only two classes (mitochondria and background), the model had more than enough learning capacity. With multiple types of objects to identify, the model will perform worse, especially if the objects are similar in appearance (e.g. mitochondria and cell nucleus).
2. Usage of instance segmentation over semantic segmentation. I found that the results of instance segmentation were better than that of semantic segmentation, even though instance segmentation is a more complex problem. I think this result is because Mask R-CNN's creation of a bounding box around instances of mitochondria simplifies the work of the segmentation model, since the segmentation model is pointed to a specific portion of the image to do its work. As a result, the segmentation in Mask R-CNN is actually more precise than in FCN.

## Future Work
Some ideas for future experimentation are:
* Increasing dataset and data augmentation techniques. The dataset I trained on was very small compared to the datasets used in deep learning problems. The model certainly has the capacity to learn from more data.
* Hyperparameter tuning. Adjusting the learning rate, batch size, epochs, and Mask R-CNN specific parameters will lead to improvements on training results.
* Testing on more objects in the image. Although I trained only on mitochondria, there are many more objects of interest in biological microscope images.
* Deploying to users. Using infrastructure to host and continually train the model online while providing a service to biology researchers.
