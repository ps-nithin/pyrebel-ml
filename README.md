# pyrebel-ml

Machine learning using an alternative approach without using neural nets. This method uses data abstraction for machine learning.

The program consists of three parts
1. Forming an abstract representation of data which is used to obtain meaningful information from the input.
2. Learning. Creating a knowledge base based on the abstract information obtained.
3. Recognition. Using the abstract data stored in knowledge base to recognize novel inputs.

The program is implemented in python and tested to run on nvidia jetson nano / orin nano devices. The program uses numba python library for using cuda.

# Learning 
Patterns and figures have a definite boundary. Each boundary is represented by a signature. Not a single signature but layers of signatures ranging from the most abstract to the least abstract representaion. 

Running the following command reads the input file to obtain signatures upto the layers specified and writes it into the knowledge base.

```python3 pynvrebel.py --learn <filename.png> or <path/to/learn/> --layer <layers>```

For example, running the command ```python3 pynvrebel.py --learn images/c.png --layer 20``` learns the file c.png and links the signatures with the symbol "c.png".
<br>Similarly, running the command ```python3 pynvrebel.py --learn images/letters_standard/ --learn 20``` learns all the files in the directory images/letters_standard/.

# Recognition
Layers of signatures is obtained for the input data. It is then compared to known signatures in the knowledge base to identify the pattern.

Running the following command reads the input file, obtains signatures upto the layers specified and compares it with the knowledge base to identify any learned patterns. The recognized symbols are then displayed.

```python3 pynvrebel.py --recognize <filename.png> --layer <layers>```

For example, running the following command
```python3 pynvrebel.py --recognize images/c_ripple.png --layer 20```
the program checks the signatures in the input image with the knowledge base and displays the recognized symbols, if any.

# Read more about the logic implemented <a href="https://github.com/ps-nithin/pyrebel-ml/blob/main/abstract/intro-r1.pdf">here</a>
