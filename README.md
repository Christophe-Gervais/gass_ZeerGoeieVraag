# Zeer Goeie Vraag?

## Installation

`pip install -r requirements.txt`

- You will need to add a folder yourself called "videos", and put in here the videos you will use.

## How to run training?

1. Go to train.py
2. If you have cuda, dont do anything.
3. If you don't, delete line 17. And remove the "," after line 16
4. Change workers on line 16 for each system. My system can do 20 workers, yours can maybe do 10 or less.
5. After all this, you can run train.py

## How to detect and trace in video?

1. Go to detect.py
2. Change video path if needed
3. Run detect.py