- What is actually required here?-
    Refill Bottles -> OK = Correct bottles
                     NOK = Dents, scratches, dirty, wrong brand, wrong bottle type, ... 
        Check bottle weight
        Check expiration date

- How to solve?-
    
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
    

- How to evaluate the performance?
    - Measure milliseconds taken to go from image to output.


- How to keep things practical? (fast development)
    - Having clear goals/tasks


- How to see progress? (Website, application, ...)
    - Keep a separate collection of images that are not fed into the model during training. Then inference on these and calculate percentage of correct responses from the model.