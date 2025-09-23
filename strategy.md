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
    2. Two models:
        - Classifying dents, dirt and bottle type
        - Text recognition
    3. Three models:
        1. Downscale input image
        2. Highlight interest areas
        3. Crop areas and inference text and classification on these
    
    - Suggested architecture:

    Two models, but initally only the classification one:
    Replacing the first job is easy as AI is very good at classification out of the box. It will definitely be very fast and capable at detecting damaged bottles.

    The text recognition is much more difficult but is also possible. I suggest making the text detection model generate a certainty value, if that is below a threshold there are two options to deal with the gas cylinder:
    1. Discard bottle in case of uncertainty.
    2. Separate bottles that pass classification but fail text to a separate batch and have one employee do a manual check on these remaining edge cases for which the model can't guarantee accuracy.



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
2. Manually classify them into training data
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

- Type of defect detected and rect of the area where this is detected
    1. Dent/deformation/wrong model
    2. Dirt
