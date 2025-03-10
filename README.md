# Speed Sign Board Detection and Speed Control System

Directions to run the code
1. Unzip the whole repository and make it your current directory 
2. Install all the required dependencies using teh requirments.txt file
    * if you are using windows machine
    * **Type** - `pip install -r requirements.txt`
    * Or if you are using anaconda prompt
    * **Type** - `conda install --file requirements.txt`
3. For running on live camera feed 
    * **Type** - `python detect.py --source usb0`
    * Or if you're using Raspberry Pi
    * **Type** - `python detect.py --model speed_sign_board.pt --source picamera0 --resolution 640x840`
