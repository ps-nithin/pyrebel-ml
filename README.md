# pyrebel
# On Jetson Nano
# Learning
Usage:
```python3 pynvrebel.py --input <filename.png> --layer <layers> --learn <symbol name>```<br>
For example, 
```python3 pynvrebel.py --input images/c.png --layers 16 --learn "c"```<br>

# Recognition
Usage:
```python3 pynvrebel.py --input <filename.png> --layer <layers> --recognize 1```<br>
For example, 
```python3 pynvrebel.py --input images/c_ripple.png --layers 16 --recognize 1```<br>
# Note
Input image file for learning should contain a single blob or len(nz_s) should be equal to 3.

# Read more about the logic implemented <a href="https://github.com/ps-nithin/pyrebel/blob/main/abstract.pdf">here</a>

# Let the data shine!
