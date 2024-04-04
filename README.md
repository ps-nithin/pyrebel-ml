# pyrebel-ml
# On Jetson Nano
# Learning
Usage:
```python3 pynvrebel.py --input <filename.png> --layer <layers> --learn <symbol name>```<br>
For example, 
```python3 pynvrebel.py --input images/c.png --layer 16 --learn "c"```<br>

# Recognition
Usage:
```python3 pynvrebel.py --input <filename.png> --layer <layers> --recognize 1```<br>
For example, 
```python3 pynvrebel.py --input images/c_ripple.png --layer 16 --recognize 1```<br>
# Note
1. Input image file for learning should contain a single blob of what is to be learned.
2. Learned knowledge is stored in 'know_base.pkl'. Delete 'know_base.pkl' to reset learning.
# Read more about the logic implemented <a href="https://github.com/ps-nithin/pyrebel/blob/main/abstract.pdf">here</a>

# Let the data shine!
