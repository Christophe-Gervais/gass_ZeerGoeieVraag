- What is actually required here?
    Refill Bottles -> OK = Correct bottles
                     NOK = Dents, scratches, dirty, wrong brand, wrong bottle type, ... 
        Check bottle weight
        Check expiration date

- How to solve?
    
    - Training data

    - Literature?
        Documenting? 
        Tests
        
    - Existing products?
        - [YOLO](https://docs.ultralytics.com/models/yolo11/)
        
    - Own Algorithm?
        - Fine tune of existing models. (transfer learning)


    - Potential architectures:
    
    1. Single model: Image -> yes/no
    2a. Two models:
        - Classification for dents, dirt and bottle type - Simply take the images of bad and good and let the AI figure out what makes one good or bad.
        - Text recognition
    2b. Two models:
        - Detection for dents, dirt and bottle type - Teach the AI to draw a rect around dents, dirt and other errors, this allows it to say what's wrong with it and should be more accurate at cost of inference and training + data prep time.
        - Text recognition
    3. Three models:
        1. Downscale input image
        2. Highlight interest areas
        3. Crop areas and inference text and detection on these
    
    - Suggested architecture:

    Two models, but initally only one and best to start with classification to ease dataset prep:
    Replacing the first job is easy as AI is very good at classification. It will definitely train and inference very fast and be capable at detecting damaged bottles.
    If this works, accuracy can be increased with larger, cleaner dataset, larger model, more training runs or switch to a detection model.

    We can also split our group, have two people prepare a classification model and dataset and two others can try a detection model and we can try to beat each other at accuracy, then we pick the winning model to show off in presentation.

    The text recognition is much more difficult but is also possible. I suggest making the text detection model generate a certainty value, if that is below a threshold there are two options to deal with the gas cylinder:
    1. Discard bottle in case of uncertainty.
    2. Separate bottles that pass detection but fail text to a separate batch and have one employee do a manual check on these remaining edge cases for which the model can't guarantee accuracy.



- How to evaluate the performance?
    - Measure milliseconds taken to go from image to output.

    From my experience the YOLO11 models easily run realtime on Nvidia hardware and can even run realtime on Intel iGPUs or edge devices by using Intel's own OpenVINO runtime which seems to perform about twice as fast on Intel hardware (60fps from nano up to medium, maybe even larger models).


- How to keep things practical? (fast development)
    - Having clear goals, tasks and steps
    - Distribution of the async tasks between teammembers
    - Start off with tiny model, scale up once that starts to work


- How to see progress? (Website, application, ...)
    - Keep a separate collection of images that are not fed into the model during training. Then inference on these and calculate percentage of correct responses from the model.


# Suggested Steps

1. Collect sample images
2. Manually split them into training data
2. Write the fine tune script
2. Write the inference script
3. Fine tune the model
4. Inference the model on remaining images
5. Manually go through the model's predicted results to validate the accuracy and correct them if necessary, once manually verified these can be added to the training set
6. Fine tune the model again on the new, extended and manually validated training set

(lines with the same number can be executed async by teammembers)

We can start testing the scripts on the [nano model](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt) for the fastest development speed. Once the training and inference pipeline is set up and we have a starting dataset we can move to a larger model to allow an increase of result accuracy.

## Image processing before inference and training

The images should best be downscaled to a small resolution for inference to save on computation. Power and time required to inference goes up exponentially with image dimensions.
I recommend starting off with 480x640

## Inference script console output requirements

The script should log how many images are processed per second and how many milliseconds the last image took and how many milliseconds the average processing per image of the current run took.

## Values we need the model to output

There are two options

1. Type of defect detected and rect of the area where this is detected. (Detection model)
    1. Dent
    2. Deformation/wrong model
    3. Dirt

2. Which class the image belongs to. This can be a class for good and a class for bad, or more detailed classes like good, bad because dirty, bad because dented or bad because wrong model. (Classification model)



## Potential accuracty and performance improvements

Adding a bright light at an angle and using multiple cameras from different angles or one camera and one light but make the gas tank rotate to get a full 360 view of it. 

Having a light shine onto the tanks sideways should allow you to see any dents much easier as they would make a higher contrast shadow that would be very easy for an AI model to detect.

It would be possible to use one or more iPhones/Xbox Kinects to generate a depth map of the gas tank using their PrimeSense cameras.
This would allow a detailed and accurate 3D model to be generated of the gas tank and thus detect dents, dirt or other uneven parts in the surface that aren't easily visible to a camera.