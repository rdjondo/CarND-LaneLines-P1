#**Finding Lane Lines on the Road** 
<img src="laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  


**To setup the project, check out the original notes at the end of this README**

#**My solution for Image segmentation on the challenge video**
My solution for the Udacity challenge for the firt project. 
Notice how the colour of the tarmac changes on the bridge to become very close to the colour of the left lane.
https://www.youtube.com/watch?v=-x-1rLKzM0I

I used some colour segmentation and this turned out to be very successful for this project.    
Some image segmentation techniques that I found very useful for this project.
This technique was particularly useful for the challenge.


<img src="my_images/original_frame.png" width="480" alt="original_frame" />

    # Let's get the yellow in the image
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # In OpenCV the HSV space is bounded by 0 and 180 on H, 0 and 255 on S, 0 and 255 on V   
    lower_yellow = np.array([18, 50, 50], np.uint8)
    upper_yellow = np.array([30, 255, 255], np.uint8)
    
    # Calculate mask for color segmentation
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    im_yellow = cv2.bitwise_and(img,img, mask= mask)
    
    plt.imshow(im_yellow)
    plt.title('Image masked for getting yellow pixels')
    plt.show()

<img src="my_images/yellow_masked.png" width="480" alt="original_frame" />

    # Let's get the white from the image
    lower_white = np.array([150, 150, 190], np.uint8)
    upper_white = np.array([255, 255, 255], np.uint8)
    mask = cv2.inRange(img, lower_white, upper_white)
    im_white = cv2.bitwise_and(img,img, mask= mask)
    
    plt.imshow(im_white)
    plt.title('Image masked for getting white pixels')
    plt.show()
    

<img src="my_images/white_masked.png" width="480" alt="original_frame" />
    
    # Combine selected images
    img = np.clip(im_white.astype('uint32')+ im_yellow.astype('uint32'),0,255).astype('uint8')
      
    plt.imshow(img)
    plt.title('Combining yellow and white segmented images')
    plt.show()

<img src="my_images/combining_yellow_white.png" width="480" alt="original_frame" />

    # Convert image to grascale
    #gray = grayscale(img.astype('uint8')).astype('int32')
    gray = grayscale(img.astype('uint8'))
    
    gray_masked = region_of_interest(gray,vertices)
    
    # Filter image (blur/low pass filter)
    kernel_size = 7
    gray = gaussian_blur(gray, kernel_size)
    
    
    # Edge detector
    low_threshold = 50
    high_threshold = low_threshold * 3
    gray = canny(gray, low_threshold, high_threshold)
    
    plt.imshow(gray, cmap='gray')
    plt.title('Edge detection on segmented image')
    plt.show()

<img src="my_images/edge_detection.png" width="480" alt="original_frame" />

    # Mask road surface
    gray = region_of_interest(gray,vertices)
    plt.imshow(gray, cmap='gray')
    plt.title('Masking road surface')
    plt.show()
    

<img src="my_images/edge_masked.png" width="480" alt="original_frame" />

    

#**Original Notes for the project** 
**Step 1:** Getting setup with Python

To do this project, you will need Python 3 along with the numpy, matplotlib, and OpenCV libraries, as well as Jupyter Notebook installed. 

We recommend downloading and installing the Anaconda Python 3 distribution from Continuum Analytics because it comes prepackaged with many of the Python dependencies you will need for this and future projects, makes it easy to install OpenCV, and includes Jupyter Notebook.  Beyond that, it is one of the most common Python distributions used in data analytics and machine learning, so a great choice if you're getting started in the field.

Choose the appropriate Python 3 Anaconda install package for your operating system <A HREF="https://www.continuum.io/downloads" target="_blank">here</A>.   Download and install the package.

If you already have Anaconda for Python 2 installed, you can create a separate environment for Python 3 and all the appropriate dependencies with the following command:

`>  conda create --name=yourNewEnvironment python=3 anaconda`

`>  source activate yourNewEnvironment`

**Step 2:** Installing OpenCV

Once you have Anaconda installed, first double check you are in your Python 3 environment:

`>python`    
`Python 3.5.2 |Anaconda 4.1.1 (x86_64)| (default, Jul  2 2016, 17:52:12)`  
`[GCC 4.2.1 Compatible Apple LLVM 4.2 (clang-425.0.28)] on darwin`  
`Type "help", "copyright", "credits" or "license" for more information.`  
`>>>`   
(Ctrl-d to exit Python)

run the following commands at the terminal prompt to get OpenCV:

`> pip install pillow`  
`> conda install -c https://conda.anaconda.org/menpo opencv3`

then to test if OpenCV is installed correctly:

`> python`  
`>>> import cv2`  
`>>>`  
(Ctrl-d to exit Python)

**Step 3:** Installing moviepy  

We recommend the "moviepy" package for processing video in this project (though you're welcome to use other packages if you prefer).  

To install moviepy run:

`>pip install moviepy`  

and check that the install worked:

`>python`  
`>>>import moviepy`  
`>>>`  
(Ctrl-d to exit Python)

**Step 4:** Opening the code in a Jupyter Notebook

You will complete this project in a Jupyter notebook.  If you are unfamiliar with Jupyter Notebooks, check out <A HREF="https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python" target="_blank">Cyrille Rossant's Basics of Jupyter Notebook and Python</A> to get started.

Jupyter is an ipython notebook where you can run blocks of code and see results interactively.  All the code for this project is contained in a Jupyter notebook. To start Jupyter in your browser, run the following command at the terminal prompt (be sure you're in your Python 3 environment!):

`> jupyter notebook`

A browser window will appear showing the contents of the current directory.  Click on the file called "P1.ipynb".  Another browser window will appear displaying the notebook.  Follow the instructions in the notebook to complete the project.  
