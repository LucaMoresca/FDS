#!/usr/bin/env python
# coding: utf-8

# # **Fundamentals of Data Science - Winter Semester 2024**
# 
# 
# ##### Matteo Migliarini (TA), Matteo Rampolla (TA) and Prof. Indro Spinelli
# <migliarini.1886186@studenti.uniroma1.it>, <rampolla.1762214@studenti.uniroma1.it>, <spinelli@di.uniroma1.it>
# 
# ---
# 
# ## **#1 Homework: Image Filtering and Object Identification**

# This first homework covers the main topics discussed in class.
# Initially, you will be dealing with the original **Image Filtering** methods (*Question 1*), then you will have to work with **Edge Detection** algorithms (*Question 2*), and lastly, you are going to operate first-hand on **Object Identification** techniques (*Question 3*).
# Additionally, the homework will contain written questions and reports where you will be asked to write down your answer in Markdown language.
# The maximum possible grade is 30, but notice that you have 5 extra bonus points scattered across the homework to further boost your grade.
# 
# *Note: your task is to fill in the missing code where you see `"YOUR CODE HERE"` and the text part `"WRITE YOU ANSWER HERE"` part corresponding to each subproblem and produce brief reports on the results whenever necessary.*
# 
# As part of the homework, provide the answer to questions in this notebook report-like manner. After you have implemented all the missing code in the required sections, you will be able to run all the code without any errors. We kindly ask you to double-check this since **all** the delivered homework will be executed.
# 
# The completed exercise should be handed in as a single notebook file. Use Markdown to provide equations. Use the code sections to provide your scripts and the corresponding plots.
# 
# -------------------------------------
# 
# **Submit it** by sending an email to:
# 
#  **<migliarini.1886186@studenti.uniroma1.it>, <rampolla.1762214@studenti.uniroma1.it> and <spinelli@di.uniroma1.it> by Sunday 27th October 23:59**.
# 
# -------------------------------------

# **Outline and Scores for #1 Homework:**
# 
# 
# * **Question 1: Image Filtering** *(7 points)*
#   * **Question 1.0: Warm Up** *(1 point)*
#   * **Question 1.1: 1D Filters** *(3 points)*
#   * **Question 1.2: 2D Filters** *(3 points)*
# 
# * **Question 2: Multi-Scale Image Representations** *(12 points)*
#   * **Question 2.1: Prewitt Operator** *(2 point)*
#   * **Question 2.2: Canny Edge Detector and Template Matching** *(4 points)*
#   * **Question 2.3: Harris Corner Detector** *(2 points)*
#   * **Question 2.4: Gaussian Pyramid and Aliasing** *(2 points)*
#   * **Question 2.5: Multi-Scale Template Matching** *(2 points)*
# 
# * **Question 3: Object Identification** *(10 points)*
#   * **Question 3.1: 3D Joint Color Histogram** *(2 points)*
#   * **Question 3.2: Types of Histograms** *(2 points)*
#   * **Question 3.3: Histogram Metrics** *(2 points)*
#   * **Question 3.4: Image Retrieval** *(2 points)*
#   * **Report** *(2 points)*
# 
# * **Question 4: Performance Evaluation** *(6 points)*
#   * **Question 4.1: Performance Evaluation** *(2 point)*
#   * **Question 4.2: Closest Neighbours** *(2 points)*
#   * **Question 4.3: Retrieval Metrics** *(1 points)*
#   * **Question 4.4: Analysis and Report** *(1 point)*
# 
# 
# **TOTAL POINTS ARE 35, MAX GRADE IS 30**
# 

# ## **Question 1: Image Filtering *(8 Points)***

# In[1]:


# Set to True the first time you run this file to install the required packages
install_packages = True
if install_packages:
    get_ipython().run_line_magic('pip', 'install pandas pillow numpy scipy matplotlib opencv-python scikit-image scikit-learn')


# In[2]:


## import packages
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import math
from scipy.ndimage import convolve1d
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from typing import List, Tuple, Optional, Dict, Union

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression

plt.rcParams["figure.figsize"] = (20, 8)


# In[3]:


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Converts RGB images to grayscale.

    Args:
        rgb (np.ndarray): RGB image.

    Returns:
        gray (np.ndarray): Grayscale image.
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def plot_pictures(
    imgs: List[np.ndarray],
    xlabels: List[str],
    nrows: int,
    ncols: int,
    show: bool = True,
    cmap: Union[str, List[str]] = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    same_scale: bool = False,
) -> List[plt.Axes]:
    """
    Plots images in a grid.

    Args:
        imgs (list[np.ndarray]): List of images.
        xlabels (list[str]): List of xlabels.
        nrows (int): Number of rows.
        ncols (int): Number of columns.
        show (bool): Whether to show the plot.
        cmap (Union[str, List[str]]): Color map.
        vmin (float): Minimum value.
        vmax (float): Maximum value.
        same_scale (bool): Whether to use the same scale for all images.

    Returns:
        axs (list[matplotlib.axes._subplots.AxesSubplot]): List of axes.
    """
    fig, axs = plt.subplots(nrows, ncols,
                            width_ratios=[img.shape[1] / img.shape[0] for img in imgs],
                            sharey=same_scale,
                            figsize=(ncols * 4, nrows * 4))
    axs = axs.flatten()

    for i, (img, ax) in enumerate(zip(imgs, axs)):
        if isinstance(cmap, list):
            current_cmap = cmap[i]
        else:
            current_cmap = cmap
        im = ax.imshow(img, cmap=current_cmap, vmin=vmin, vmax=vmax)


        ax.set_title(xlabels[i])
        ax.axis('off')
    plt.tight_layout()

    if show:
        plt.show()
    else:
        return axs


# #### **1.0: Warm Up *(1 points)***

# ##### **Convolution**

# The convolution is defined as $f[m,n]=I \ast g= \sum_{k,l} I[m-k,n-l]g[k,l]$ and is a linear operator, thus the following properties hold:
# * *Homogeneity*: $T[aX]=aT[X]$
# * *Additivity* : $T[X_1+X_2]=T[X_1]+T[X_2]$
# * *Superposition*: $T[aX_1+bX_2]=aT[X_1]+bT[X_2]$
# 
# Prove mathematically the additivity of convolution.

# -------------------------------------------------------
# 
# **WRITE YOU ANSWER HERE**
# 
# Write it in Markdown, you are encouraged to take a look at the cells in this notebook.
# You can find a cheat sheet [here](https://www.markdownguide.org/basic-syntax)
# 
# -------------------------------------------------------

# **Exercise on convolutions**
# 
# Usually, to be consistent with the definition, we use zero padding: this practice consists of adding a border of zero-valued pixels all around the edges of the input image.
# In our case, we will have
# 
# \begin{equation}
# f[m,n] =
# \begin{pmatrix}
# 0 & 0 & 0 & 0 & 0 \\
# 0 & 4 & 5 & 7 & 0 \\
# 0 & 7 & 3 & 1 & 0 \\
# 0 & 9 & 2 & 2 & 0 \\
# 0 & 0 & 0 & 0 & 0 \\
# \end{pmatrix}
# *\begin{pmatrix}
# -1 & 0 & 1 \\
# -1 & 0 & 1 \\
# -1 & 0 & 1 \\
# \end{pmatrix}
# =\begin{pmatrix}
# ... & ... & ... \\
# ... & -10 & ... \\
# ... & ... & ... \\
# \end{pmatrix}.
# \end{equation}
# 
# Now, given the images $I_1 \in \mathbb{N}^{5 \times 5}$, $I_2 \in \mathbb{N}^{5 \times 5}$, and the kernel $g \in \mathbb{N}^{2 \times 2}$, apply the additivity property for one convolution operation: you should expect that $T[I_1 + I_2]$ and $T[I_1]$ + $T[I_2]$ get the same result.
# Displaying explicitly the operations you performed, feel free to choose the coordinates to apply the convolution on as you like them.
# 
# \begin{equation}
#     I_1 =
#     \begin{pmatrix}
#         0 & 0 & 0 & 0 & 0 \\
#         0 & 3 & 3 & 5 & 0 \\
#         0 & 2 & 7 & 0 & 0 \\
#         0 & 0 & 2 & 3 & 0 \\
#         0 & 0 & 0 & 0 & 0 \\
#     \end{pmatrix}, \quad
#     I_2 =
#     \begin{pmatrix}
#         0 & 0 & 0 & 0 & 0 \\
#         0 & 3 & 5 & 3 & 0 \\
#         0 & 1 & 0 & 5 & 0 \\
#         0 & 2 & 6 & 8 & 0 \\
#         0 & 0 & 0 & 0 & 0 \\
#     \end{pmatrix}, \quad
#     g =
#     \begin{pmatrix}
#         -1 & 1 \\
#         -1 & 1 \\
#     \end{pmatrix}
# \end{equation}

# -------------------------------------------------------
# 
# **WRITE YOU ANSWER HERE**
# 
# Write it in Markdown, you are encouraged to take a look at the cells in this notebook.
# You can find a cheat sheet [here](https://www.markdownguide.org/basic-syntax)
# 
# -------------------------------------------------------

# ### **1.1: 1D Filters *(3 Points)***
# 

# ##### **1.1.1: 1-D Gaussian Filter**
# 
# Implement a method which computes the values of a 1-D Gaussian $G_x$ for a given standard deviation $\sigma$ and filter size $k$.:
# \begin{equation}
# G(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{x^2}{2\sigma^2}\right)
# \end{equation}
# 
# The method should return a vector $x$ of values on which the Gaussian filter is defined: integer values on the interval $\left[-k\sigma,k\sigma\right]$.
# 
# With $\sigma=3$ and $k=3$ you should get the following output:
# 
# ![](./images/1gauss.png)

# In[6]:


def gauss(sigma: int, filter_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Gaussian filter.

    Args:
        sigma (int): Standard deviation.
        filter_size (int): Filter size, be wary that it is the full size of the filter.

    Returns:
        Gx (np.ndarray): Gaussian filter.
        x (np.ndarray): Array of integer values.
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    
    return Gx, x


# --------------------------------------------
# 
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[8]:


sigma = 3
kappa = 3
Gx, x = gauss(sigma, kappa)
plt.figure(1)
plt.plot(x, Gx, ".-")
plt.show()


# **Variation of $\sigma$ and filter size**
# 
# Now we generate a signal and apply the Gauss filter with different values of $\sigma$ and filter size.
# Write down your considerations.

# In[9]:


# Generate a noisy signal with increased noise (example data)
t = np.linspace(0, 2 * np.pi, 1000)
noisy_signal = np.sin(t) + 1.0 * np.random.randn(len(t))  # Increased noise

# Define a fixed sigma value
sigma = 1.0

# Define a range of filter sizes to test
filter_sizes = [3, 7]

# Create subplots to compare the denoising results
plt.figure(figsize=(12, 8))

for i in range(len(filter_sizes)):
    # Get the Gaussian kernel and x values from the custom gauss function
    Gx, x = gauss(sigma, filter_sizes[i])

    # Convolve the noisy signal with the Gaussian filter
    denoised_signal = np.convolve(noisy_signal, Gx, mode="same")

    # Plot the noisy and denoised signals for the current filter_size
    plt.subplot(len(filter_sizes), 1, i + 1)
    plt.plot(t, noisy_signal, label="Noisy Signal")
    plt.plot(t, denoised_signal, label=f"Denoised (Filter Size {filter_sizes[i]})")
    plt.title(f"Effect of Filter Size on Denoising (Sigma={sigma})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()


# Define a fixed sigma value
sigma = [3, 10]

# Define a range of filter sizes to test
filter_sizes = 3

# Create subplots to compare the denoising results
plt.figure(figsize=(12, 8))

for i in range(len(sigma)):
    # Get the Gaussian kernel and x values from the custom gauss function
    Gx, x = gauss(sigma[i], filter_sizes)
    denoised_signal = np.convolve(noisy_signal, Gx, mode="same")

    # Plot the noisy and denoised signals for the current filter_size
    plt.subplot(len(sigma), 1, i + 1)
    plt.plot(t, noisy_signal, label="Noisy Signal")
    plt.plot(t, denoised_signal, label=f"Denoised (Sigma {sigma[i]})")
    plt.title(f"Effect of Sigma on Denoising (Filter Size={filter_sizes})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()


# -------------------------------------------------------
# 
# **WRITE YOU ANSWER HERE**
# 
# -------------------------------------------------------

# ##### **1-D Box Filter**
# 
# Implement a method that computes the values of a 1-D Box $B_x$ for a given standard deviation $\sigma$:
# \begin{equation}
# B= \left[\frac{1}{n},\frac{1}{n},\dots,\frac{1}{n}\right] \in \mathbb{R}^n
# \end{equation}
# 
# The method should also return a vector $x$ of values on which the filter is defined: integer values in the interval $\left[-3\kappa,3\kappa\right]$.
# 
# An example of a Box Filter:
# 
# ![](./images/1box.png)

# In[10]:


def box(filter_size: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Box filter.

    Args:
        filter_size (int): Filter size.

    Returns:
        Bx (np.ndarray): Box filter.
        x (np.ndarray): x values.
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################


    return Bx, x


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[12]:


## function box
size = 3
Bx, x = box(size)
plt.figure(1)
plt.plot(x, Bx, ".-")
plt.show()


# ##### **1.1.2: 1D Laplacian filter**

# Derive the 1-D Laplacian $L_x$, then implement a method that computes it for a given standard deviation $\sigma$ and filter size $k$:
# 
# With $\sigma=3$ and $k=10$ you shouls expect the following output:
# 
# ![](./images/1laplace.png)

# -------------------------------------------------------
# **WRITE YOU ANSWER HERE**
# 
# Straight answers not showing the derivation steps will not be accepted.
# 
# \begin{equation}
# \frac{d^2G(x)}{dx^2}=
# \end{equation}
# 
# -------------------------------------------------------

# In[13]:


def laplace(sigma: int, filter_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Laplace 1D filter.

    Args:
        sigma (int): Standard deviation.
        filter_size (int): Filter size.

    Returns:
        Lx (np.ndarray): Laplace filter.
        x (np.ndarray): x values.
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    return Lx, x


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[15]:


sigma = 3
filter_size = 10
Lx, x = laplace(sigma, filter_size)
plt.figure(1)
plt.plot(x, Lx, ".-")
plt.show()


# #### **1.1.3: An example of non-linear filter: 1-D median filter**

# Up until now we have worked with linear 1-D filters, you are now asked to implement a non-linear one: the 1-D median filter.
# At each position $p$ of a given 1-D signal $\mathcal{S}$, the median filter of size $s$ takes the neighborhood $\mathcal{N}_s(p)=[\mathcal{S}(p - \frac{s - 1}{2}), \mathcal{S}(p + \frac{s - 1}{2})]$, and replaces the value of $p$ with the median of the values in $\mathcal{N}_s(p)$.
# Since it removes outliers from $\mathcal{N}_s(p)$, the median filter is often used in image and signal processing to remove noise.
# 
# Complete the method below to compute the median filter $m(\cdot)$ of size $s$ for all elements of a signal $\mathcal{S}$.
# The method should return the filtered signal.
# From the description above, you may have noticed that one must *carefully* consider how to deal with the boundaries of the signal.
# In this exercise, you are free to choose the strategy to deal with the boundaries, you may look [here](https://en.wikipedia.org/wiki/Median_filter).
# 
# You should get the following input-output pair:
# 
# ![1dmedian](./images/1dmedian.png)

# In[16]:


def median_filter(signal: np.ndarray, filter_size: int) -> np.ndarray:
    """
    Median 1D filter.
    You are free to choose how to deal with the beginning and the end of the signal.
    You can use the `np.median` function https://numpy.org/doc/stable/reference/generated/numpy.median.html

    Args:
        signal (np.ndarray): 1-D array of values
        filter_size (int): size of the filter

    Returns:
        signal (np.ndarray): 1-D array of values with the median values computed at each position
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    return signal


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[18]:


filter_size = 3
signal = lambda x: 2 * np.sin(x) + np.sin(3 * x) + 0.5 * np.sin(7 * x)
x = np.linspace(0, 50, 200)
in_signal = signal(x)
out_signal = median_filter(in_signal.copy(), filter_size)
fig, axs = plt.subplots(2)
fig.suptitle("Median filter")
axs[0].plot(x, in_signal, ".-", label="$\\mathcal{S}$", color="blue")
axs[0].grid()
axs[0].legend()
axs[1].plot(x, out_signal, ".-", label="$m(\\mathcal{S})$", color="red")
axs[1].grid()
axs[1].legend()
plt.show()


# #### **1.1.4: Check linearity of the filters**

# In *Question 1.0: warmup* we have seen some of the properties of linear operators and we have proven the additivity property of convolution.
# Consider the 1-D Gaussian filter $g(\cdot)$, the 1-D Laplacian filter $l(\cdot)$, and the 1-D median filter $m(\cdot)$.
# You are given signals $\mathcal{S}_1$ and $\mathcal{S}_2$, by taking advantage of the functions you have implemented in the previous exercises, empirically verify the following:
# 
# 1. $g(\mathcal{S}_1 + \mathcal{S}_2) = g(\mathcal{S}_1) + g(\mathcal{S}_2)$
# 2. $l(\mathcal{S}_1 + \mathcal{S}_2) = l(\mathcal{S}_1) + l(\mathcal{S}_2)$
# 3. $g(l(\mathcal{S}_1 + \mathcal{S}_2)) = (g * l) * \mathcal{S}_1 + (g * l) * \mathcal{S}_2$
# 4. $m(\mathcal{S}_1 + \mathcal{S}_2) \neq m(\mathcal{S}_1) + m(\mathcal{S}_2)$
# 
# Can you explain why the median filter is not linear?

# *HINT: you can exploit the function `convolve` from `scipy.signal` to compute the convolution with the Gaussian and the Laplacian filters.*

# In[19]:


x = np.linspace(0, 50, 200)
s1 = 2 * np.sin(x) + np.sin(3 * x) + 0.5 * np.sin(7 * x)
s2 = 2 * np.cos(x) + np.cos(3 * x) + 0.5 * np.cos(7 * x)
sigma, filter_size = 5, 3

#####################################################
##                 YOUR CODE HERE                  ##
#####################################################


g_s1_s2 = "YOUR CODE FOR THE FIRST TERM OF EQ.1"
g_s1_g_s2 = "YOUR CODE FOR THE SECOND TERM OF EQ.1"
l_s1_s2 = "YOUR CODE FOR THE FIRST TERM OF EQ.2"
l_s1_l_s2 = "YOUR CODE FOR THE SECOND TERM OF EQ.2"
g_l_s1_s2 = "YOUR CODE FOR THE FIRST TERM OF EQ.3"
g_l_s1_g_l_s2 = "YOUR CODE FOR THE SECOND TERM OF EQ.3"
m_s1_s2 = "YOUR CODE FOR THE FIRST TERM OF EQ.4"
m_s1_m_s2 = "YOUR CODE FOR THE SECOND TERM OF EQ.4"


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[21]:


assert np.allclose(g_s1_s2, g_s1_g_s2), "Equation (1) is not verified"
assert np.allclose(l_s1_s2, l_s1_l_s2), "Equation (2) is not verified"
assert np.allclose(g_l_s1_s2, g_l_s1_g_l_s2), "Equation (3) is not verified"
assert not np.allclose(m_s1_s2, m_s1_m_s2), "Equation (4) is not verified"


# In[22]:


fig, axs = plt.subplots(5, 1)
fig.suptitle("Check linearity of the filters")
ys = [
    s1,
    s2,
    g_s1_s2,
    g_s1_g_s2,
    l_s1_s2,
    l_s1_l_s2,
    g_l_s1_s2,
    g_l_s1_g_l_s2,
    m_s1_s2,
    m_s1_m_s2,
]
labels = [
    "$\\mathcal{S}_1$",
    "$\\mathcal{S}_2$",
    "$g(\\mathcal{S}_1 + \\mathcal{S}_2)$",
    "$g(\\mathcal{S}_1) + g(\\mathcal{S}_2)$",
    "$l(\\mathcal{S}_1 + \\mathcal{S}_2)$",
    "$l(\\mathcal{S}_1) + l(\\mathcal{S}_2)$",
    "$g(l(\\mathcal{S}_1 + \\mathcal{S}_2))$",
    "$(g * l) * \\mathcal{S}_1 + (g * l) * \\mathcal{S}_2$",
    "$m(\\mathcal{S}_1 + \\mathcal{S}_2)$",
    "$m(\\mathcal{S}_1) + m(\\mathcal{S}_2)$",
]
colors = [
    "blue",
    "green",
    "red",
    "gray",
    "purple",
    "gray",
    "orchid",
    "gray",
    "orange",
    "gray",
]
for y_idx in range(0, len(ys), 2):
    i = y_idx // 2
    axs[i].plot(x, ys[y_idx], ".-", label=labels[y_idx], color=colors[y_idx])
    axs[i].plot(
        x, ys[y_idx + 1], "--", label=labels[y_idx + 1], color=colors[y_idx + 1]
    )
    axs[i].grid()
    axs[i].legend(loc="upper right")
fig.tight_layout()
plt.show()


# #### **1.1.5: Butterworth Filter**

# The Butterworth filter is a type of frequency domain filter characterized by a smooth transition between passband and stopband.
#     It is commonly used in signal processing to filter out frequencies beyond a certain cutoff, while maintaining a relatively flat frequency response in the passband.
#     The sharpness of the cutoff is controlled by the order of the filter (n).
# 
# \begin{equation}
# H(x) = \frac{1}{1 + \left( \frac{x}{\text{cutoff}} \right)^{2n}}
# \end{equation}
# 
# 
#  
# 

# With $\verb|cutoff|=20$ , $\verb|filtersize|=30$ and $n=2$ you shouls expect the following output:
# 
# ![1dbutterworth](./images/Butterworth_Filter.png)

# In[23]:


def butterworth_filter_1d(cutoff: float, filter_size: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements a 1D Butterworth filter.

    Args:
        cutoff (float): The frequency cutoff point. Determines the transition point between passband and stopband.
        filter_size (int): Size of the filter. The filter will have `filter_size` elements symmetrically centered around zero.
        n (int): The order of the Butterworth filter. Higher values result in a sharper transition between passband and stopband.

    Returns:
        Hx (np.ndarray): The 1D Butterworth filter coefficients.
        x (np.ndarray): The x values corresponding to the filter, centered around zero.

    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    return Hx, x


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[25]:


cutoff = 20.0
filter_size = 30
n = 2

# Generate the Butterworth filter
Hx, x = butterworth_filter_1d(cutoff, filter_size, n)

# Plot the Butterworth filter
plt.figure(1)
plt.plot(x, Hx, ".-")
plt.title("1D Butterworth Filter")
plt.xlabel("x")
plt.ylabel("Hx")
plt.show()


# ### **1.2: 2D Filters *(3 Points)***
# 
# 

# ##### **2-D Laplacian filter**
# 
# The function should take an image as input and return the result of the convolution of the image with a 2-D Laplacian kernel.
# You can use Python’s `convolve2D` function if you don’t want to implement it yourself.
# 
# See the figure below for an illustration of Laplacian filtering.
# 
# ![](./images/stella_laplace.jpg)

# In[26]:


def laplacefiltering2D(img: np.ndarray, sigma: int, filter_size: int = 3) -> np.ndarray:
    """
    Implement a 2D Laplacian filter.

    Args:
        img (np.ndarray): Input image.
        sigma (int): Standard deviation.
        filter_size (int): Filter size.

    Returns:
        smooth_img (np.ndarray): Smoothed image.
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    return smooth_img


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[28]:


img = rgb2gray(np.array(Image.open("./images/stella.jpg")))
smooth_img = laplacefiltering2D(img, 3)

imgs = [img, smooth_img]
labels = ["Input Image", "Laplacian Filtered Image"]

plot_pictures(imgs, labels, nrows=1, ncols=2, cmap="gray", vmin=None, vmax=None)


# ##### **2-D Box filter**
# 
# The function should take an image as input and return the result of the convolution of this image with a 2D Box kernel.
# Beware of errors in the following code, find them, and correct them.
# You will be provided with the image to which the correct box filter has already been applied and you can use the SSIM metric to check that your result is correct, $SSIM \approx 1$ means that it is correct.
# 
# See the figure below for an illustration of Box filtering.
# 
# ![](./images/box_image.jpg)

# In[29]:


def boxfiltering(img: np.ndarray, filter_size: int) -> np.ndarray:
    """
    Implement a 2D Box filter.

    Args:
        img (np.ndarray): Input image.
        filter_size (int): Filter size.

    Returns:
        smooth_img (np.ndarray): Smoothed image.

    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    

    return smooth_img


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[31]:


img = rgb2gray(np.array(Image.open("./images/stella.jpg")))
smooth_img = np.load("assets/box_filter.npy")
smooth_fooled = boxfiltering(img, 20)


imgs = [img, smooth_img, smooth_fooled]
labels = ["Input Image", "Box Filtered Image", "Box Filtered Image (Yours)"]

plot_pictures(imgs, labels, nrows=1, ncols=3, cmap="gray", vmin=None, vmax=None)

print(
    f"target_ssim_score:\t{ssim(smooth_img, smooth_img, data_range=smooth_img.max() - smooth_img.min())}\nyour_ssim_score:\t{ssim(smooth_img, smooth_fooled, data_range=smooth_img.max() - smooth_img.min())}"
)


# 
# ##### **Separability and computational efficiency**

# Analitically prove that the Laplacian filter is separable.
# Recall that the Laplacian operator is defined as $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$.

# -------------------------------------------------------
# 
# 
# **WRITE YOU ANSWER HERE**
# 
# 
# -------------------------------------------------------

# Briefly explain why it is more convenient computationally speaking to use two 1D Laplacian filters rather than one 2D Laplacian filter.
# Assume that the dimension of the 2D Laplacian filter is $k^2$ and that the dimension of the two 1D Laplacian filters is $k$.

# -------------------------------------------------------
# 
# 
# **WRITE YOU ANSWER HERE**
# 
# 
# -------------------------------------------------------

# ##### **Practical example of separability**

# Up to this point, we have seen that both the Gaussian and the Laplacian operators are linear and you have explicitly proven that the Laplacian operator is separable.
# As you may guess, the Gaussian operator is also separable, and so is the composition of them: the Laplacian of Gaussian (LoG) operator.
# Recall from section section "2-D Laplacian filter" that the LoG operator is defined as:
# 
# \begin{equation}
# \nabla^2 \circ G = \frac{d^2G(x)}{dx^2}=-\frac{\sigma^2 - x^2}{\sqrt{2\pi}\sigma^5}\exp\left(-\frac{x^2}{2\sigma^2}\right)
# \end{equation}
# 
# Previously, you have implemented a 2-D LoG filter without leveraging the separability property.
# In this section, you're asked to implement again the 2-D LoG filter **exploiting the separability**, i.e. convolving the image with a separable 1-D Laplacian kernel along axes.
# 
# Compare the images to check that the separability property is valid. Note that the last image shows the differences between the separable case and the non separable case: it should be black up to numerical approximations, i.e. it should cointain approximately 0 everywhere.

# In[32]:


def laplacefiltering(img: np.ndarray, sigma: int, filter_size: int = 3) -> np.ndarray:
    """
    Implement a 2D Laplacian filter, leveraging the previous laplacian and the separability property.

    Args:
        img (np.ndarray): Input image.
        sigma (int): Standard deviation.
        filter_size (int): Filter size.

    Returns:
        derived_img (np.ndarray): Derived image.
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    return img_x


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[34]:


img = rgb2gray(np.array(Image.open("images/stella.jpg")))
derived_img_separable = laplacefiltering(img, sigma=3)
derived_img_non_separable = laplacefiltering2D(img, sigma=3)
difference_img = derived_img_separable - derived_img_non_separable

plt.figure(2)
ax1 = plt.subplot(1, 4, 1)
ax1.set_xlabel("Input Image")
ax2 = plt.subplot(1, 4, 2)
ax2.set_xlabel("Laplacian Filtered Image (separable case)")
ax3 = plt.subplot(1, 4, 3)
ax3.set_xlabel("Laplacian Filtered Image")
ax4 = plt.subplot(1, 4, 4)
ax4.set_xlabel("Difference image (separable - non separable)")


plt.sca(ax1)
plt.imshow(img, cmap="gray")
plt.sca(ax2)
plt.imshow(derived_img_separable, cmap="gray")
plt.sca(ax3)
plt.imshow(derived_img_non_separable, cmap="gray")
plt.sca(ax4)
plt.imshow(difference_img * 100, cmap="gray", vmin=0, vmax=255)
plt.show()

assert ssim(derived_img_separable, derived_img_non_separable, data_range=derived_img_separable.max() - derived_img_separable.min()) > 0.999, "The separable and non-separable Laplacian filters are not the same."


# ##### **Gaussian Blur Filter**
# 
# Implement a 2D Gaussian filter.
# Make it so that the filter size changes adaptively with the size of sigma, and round it up to the nearest odd integer.
# $$
# \verb|filter_size| = \lceil 3\sigma \rceil
# $$
# 
# See the figure below for an illustration of the blurring filter:
# 
# ![](./images/moon_gauss.jpg)

# In[35]:


def gaussfiltering(img: np.ndarray, sigma: int) -> np.ndarray:
    """
    Implement a 2D Gaussian filter, leveraging the previous gauss.

    Args:
        img (np.ndarray): Input image.
        sigma (int): Standard deviation.

    Returns:
        smooth_img (np.ndarray): Smoothed image.
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    

    return out


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[37]:


img = rgb2gray(np.array(Image.open("images/moon.jpg")))
imgs = [img, gaussfiltering(img, 10)]
labels = ["Input Image", "Harris Corner Detector Image"]
plot_pictures(imgs, labels, nrows=1, ncols=2, cmap="gray", vmin=0, vmax=255)


# #### **Discrete Laplacian Operator** 
# 
# Write the [discrete Laplacian filter](https://en.wikipedia.org/wiki/Discrete_Laplace_operator), which is a discrete approximation of the Laplacian Filter, speciic for image processing.
# 
# Now, given an image $x$, a Gaussian filter $G(x)$ and a discrete Laplacian filter $L(x)$, compute:
# - $ G(L(x)) $
# - $ L(G(x)) $
# 
# And check that the $SSIM = 1$
# 
# See the figure below for illustration of the discrete Laplacian:
# 
# ![](./images/laplacian.png)

# In[38]:


def discrete_laplace(img: np.ndarray) -> np.ndarray:
    """
    Discrete Laplacian operator.

    Args:
        img (np.ndarray): Input image.

    Returns:
        Lap (np.ndarray): Image after applying the Laplacian operator.
    """
    
    return Lap


img = rgb2gray(np.array(Image.open("./images/lenna.jpg")))
sigma = 1.0

image_edges = "YOUR CODE HERE"
image_edges_smoothed = "YOUR CODE HERE"

image_smoothed = "YOUR CODE HERE"
image_smoothed_edges = "YOUR CODE HERE"


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[40]:


# LoG
imgs = [img, image_edges*5, image_edges_smoothed*10]
labels = [
    "Input Image",
    "Image with Laplacian Operator",
    "Image with Laplacian-Gaussian Operator",
]
plot_pictures(imgs, labels, nrows=1, ncols=3, cmap="gray", vmin=0, vmax=255)

# GoL
imgs = [img, image_smoothed, image_smoothed_edges*10]
labels = [
    "Input Image",
    "Image with Gaussian Operator",
    "Image with Gaussian-Laplacian Operator",
]
plot_pictures(imgs, labels, nrows=1, ncols=3, cmap="gray", vmin=0, vmax=255)


assert ssim(
            image_edges_smoothed,
            image_smoothed_edges,
            data_range=image_edges_smoothed.max() - image_edges_smoothed.min(),
        ) > 0.9, "The LoG and GoL operators are not the same."


# ## **Question 2: Multi-Scale Image Representations** *(9 Points)*
# 
# Edges represents object boundaries, thus edge detection is a very important preprocessing step for any object detection or recognition process.
# Simple edge detection kernels are based on approximation of gradient images.
# 
# You will use some basic edge detection kernels.

# ### **2.1: Prewitt Operator** *(2 points)*
# The [Prewitt operator](https://en.wikipedia.org/wiki/Prewitt_operator) is used in image processing, particularly within edge detection algorithms.
# Technically, it is a discrete differentiation operator, computing an approximation of the gradient of the image intensity function.
# 
# See figure below for illustration of Prewitt operators in action:
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/3/3f/Bikesgray.jpg" width="300"/> <img src="https://upload.wikimedia.org/wikipedia/commons/3/3e/Bikesgray_prewitt.JPG" width="300"/>
# 
# Now implement the Prewitt operators. Try to take advantage of the separability of this filter.

# In[41]:


def prewitt_x(img: np.ndarray) -> np.ndarray:
    """
    Implement a 3x3 Prewitt discrete operator for vertical edges.

    Args:
        img (np.ndarray): Input image.

    Returns:
        Fx (np.ndarray): Image with vertical edges.
    """

    return out

def prewitt_y(img: np.ndarray) -> np.ndarray:
    """
    Implement a 3x3 Prewitt discrete operator for horizontal edges.

    Args:
        img (np.ndarray): Input image.

    Returns:
        Fx (np.ndarray): Image with horizontal edges.
    """

    return out


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[43]:


img = rgb2gray(np.array(Image.open("images/valve.png")))
imgs = [img, prewitt_x(img), prewitt_y(img)]
labels = ["Input Image", "Vertical Edges Detection", "Horizontal Edges Detection"]
plot_pictures(imgs, labels, nrows=1, ncols=3, cmap="gray", vmin=0, vmax=255)


# #### Gradient Magnitude
# Using the previously defined functions `prewitt_x` and `prewitt_y`, retrieve the gradient magnitude of the image.
# 
# See the figure below for an illustration of the Gradient Magnitude:
# 
# ![](./images/sobel.png)

# In[44]:


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Apply the Prewitt operator to the image and compute
    the gradient magnitude.

    Args:
        img (np.ndarray): Input image.

    Returns:
        magnitude (np.ndarray): Gradient magnitude.
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    return magnitude


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[46]:


img = rgb2gray(np.array(Image.open("images/valve.png")))
igms = [img, gradient_magnitude(img)]
labels = ["Input Image", "Gradient Magnitude"]
plot_pictures(igms, labels, nrows=1, ncols=2, cmap="gray", vmin=0, vmax=255)


# ### **2.2 Canny Edge Detector and Template Matching** *(3 points)*
# The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images.

# The Canny edge detection algorithm is composed of 5 steps:
# 
# 1. Noise reduction: apply Gaussian filter to smooth the image in order to remove the noise
# 2. Gradient calculation: find the intensity gradients of the image
# 3. Non-maximum suppression: apply gradient magnitude thresholding or lower bound cut-off suppression to get rid of spurious response to edge detection
# 4. Double threshold: apply double threshold to determine potential edges
# 5. Edge Tracking by Hysteresis: track edge by hysteresis: finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
# 
# 
# You can learn more about the Canny edge detector at the following links:
# - [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html).
# - [Canny Edge Detection Step by Step in Python — Computer Vision](https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123).
# - [What is a Canny Edge Detection Algorithm?](https://towardsai.net/p/computer-vision/what-is-a-canny-edge-detection-algorithm).

# ##### **Stage 1** - *Noise reduction*
# 
# Apply Gaussian filter to smooth the image in order to remove the noise.
# **You just need to run the code for this stage**.

# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[47]:


# STEP 1:
img = rgb2gray(np.array(Image.open("images/pinwheel.jpg")))
smoothed_img = gaussfiltering(img, sigma=1)

plot_pictures([img, smoothed_img], ["Input image", "Smoothed image"], 1, 2, vmax=255)


# ##### **Stage 2** - *Gradient calculation*
# 
# This step detects edges intensity and direction by calculating the gradient of the image through edge detection operators.
# 
# Edges correspond to a change of pixels’ intensity, the easiest way to detect it is applying filters that highlight it in both directions: horizontal ($x$) and vertical ($y$).
# 
# When the image is smoothed, the derivatives $D_x(I)$ and $D_y(I)$ w.r.t. $x$ and $y$ are calculated.
# It can be implemented by convolving $I$ with Prewitt kernels $K_x$ and $K_y$, respectively.

# Taking advantage of the functions `prewitt_x` and `prewitt_y` you have previously implemented, compute the gradient magnitude and direction of the image.
# Your function will output the magnitude $G$ and the slope $\theta$ of the gradient.
# Scale the magnitude to the range $[0,255]$.

# In[48]:


# STEP 2
def prewitt_filters(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function applies the prewitt filter to the input image in x and y direction.
    Scale' the magnitude to [0, 255] and return the magnitude and the direction of the gradient.

    Args:
        img (np.ndarray): input image

    Returns:
        G (np.ndarray): gradient magnitude
        theta (np.ndarray): gradient direction
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    return (G, theta)


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[50]:


G, theta = prewitt_filters(smoothed_img)

plot_pictures([G, theta], ["Gradient magnitude", "Gradient direction"], 1, 2)


# ##### **Stage 3** - *Non-maximum suppression*
# 
# The image is scanned along the image gradient direction and, if pixels are not part of the local maxima, they are set to zero.
# This has the effect of suppressing all image information that is not part of local maxima.

# In[51]:


def direction_scan(
    img: np.ndarray, theta: float, i:int, j:int
    ) -> List[float]:
    """
    This function quantizes the gradient directions into
    four discrete directions:
        - horizontal: theta in [0, 45) or [135, 180]
        - right diagonal: theta in [0, 45)
        - vertical: theta in [45, 135)
        - left diagonal: theta in [135, 180)
    then it returns the pixel values of the two pixels in
    the corresponding direction.

    Returns:
        neighbors List[float]: pixel values of the two neighbors
            in the corresponding direction
    """
    #####################################################
    ##            FIX THIS CODE (8 ERRORS)             ##
    #####################################################
    if (0 <= theta[i,j] < 0) or (0 <= theta[i,j] <= 0):
        neighbors = [(i, j+1), (i, j-1)]
    elif (0 <= theta[i,j] < 0):
        neighbors = [(i+1, j+1), (i-1, j-1)]
    elif (0 <= theta[i,j] < 0):
        neighbors = [(i+1, j), (i-1, j)]
    else:
        neighbors = [(i+1, j-1), (i-1, j+1)]

    N,M = img.shape
    pixels = [img[x,y] for x, y in neighbors if True]
    return pixels

def non_max_suppression(img: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    This function performs non-maximum suppression on the input image.

    Args:
        img (np.ndarray): input image
        D (np.ndarray): gradient direction

    Returns:
        Z (np.ndarray): image after non-maximum suppression
    """

    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.uint8)

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            #####################################################
            ##                 YOUR CODE HERE                  ##
            #####################################################
            pass

    return (Z / Z.max() * 255).astype(np.uint8)


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[53]:


Z = non_max_suppression(G, theta)

plot_pictures(
    [G, Z*2],
    ["Gradient magnitude", "Non-max suppressed magnitude"],
    1,
    2,
)


# ##### **Stage 4** - *Thresholding*
# 
# After application of non-maximum suppression, remaining edge pixels provide a more accurate representation of real edges in an image.
# However, some edge pixels remain that are caused by noise and color variation.
# To account for these spurious responses, it is essential to filter out edge pixels with a weak gradient value and preserve edge pixels with a high gradient value.
# 
# This is accomplished by selecting high and low threshold values:
# - If an edge pixel’s gradient value is higher than the high threshold value, it is marked as a strong edge pixel.
# - If an edge pixel’s gradient value is smaller than the high threshold value and larger than the low threshold value, it is marked as a weak edge pixel.
# - If an edge pixel's gradient value is smaller than the low threshold value, it will be suppressed.

# Implement the function `threshold` that takes as input the image, the low and high threshold ratios and returns a tuple whose elements are the thresholded image, the value of the weak edges and the value of the strong edges. Set the value of weak edges to 100 and the value of strong edges to 255.

# In[54]:


# STEP 4
def threshold(
    img: np.ndarray,
    low_threshold: int = 100, high_threshold: int = 200,
    weak_value :int = 100, strong_value: int = 255
) -> np.ndarray:
    """
    This function applies a double thresholding to the input image.

    Args:
        img (np.ndarray): input image
        low_threshold (int): low threshold
        high_threshold (int): high threshold
        weak_value (int): value for weak edges
        strong_value (int): value for strong edges

    Returns:
        res (np.ndarray): thresholded image
    """

    i_height, i_width = img.shape
    res = np.zeros((i_height, i_width), dtype=np.uint8)
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    return res


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[56]:


weak_value = 100
strong_value = 255
res = threshold(Z, low_threshold=25, high_threshold=100, weak_value=weak_value, strong_value=strong_value)

plot_pictures(
    [Z, res],
    ["Non-max suppressed magnitude", "Double-thresholded magnitude"],
    1,
    2,
)


# ##### **Stage 5** - *Hysteresis*
# 
# Based on the threshold results, the hysteresis consists of transforming weak pixels into strong ones, if and only if at least one of the pixels around the one being processed is a strong one.

# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[57]:


# STEP 5
def hysteresis(img: np.ndarray, weak: int = 100, strong: int = 255) -> np.ndarray:
    """
    This function applies hysteresis to the input image.

    Args:
        img (np.mdarray): input image.
        weak (int): weak threshold.
        strong (int): strong threshold.

    Returns:
        img (np.ndarray): image after hysteresis.
    """
    M, N = img.shape
    img_ = img.copy()
    strong_edges_idxs = np.argwhere(img == strong).tolist()
    while len(strong_edges_idxs) > 0:
        i, j = strong_edges_idxs.pop()
        if (i - 1 >= 0) and (j - 1 >= 0) and (i + 1 < M) and (j + 1 < N):
            weak_edges_mask = img_[i - 1 : i + 2, j - 1 : j + 2] == weak
            if np.any(weak_edges_mask):
                img_[i - 1 : i + 2, j - 1 : j + 2][weak_edges_mask] = strong
                weak_edges_idxs = (
                    (np.argwhere(weak_edges_mask) - 1) + np.array([i, j])
                ).tolist()
                strong_edges_idxs = strong_edges_idxs + weak_edges_idxs[::-1]
    img_[img_ != strong] = 0
    return img_


# In[58]:


final_img = hysteresis(res, weak=weak_value)

plot_pictures(
    [res, final_img],
    ["Double-thresholded magnitude", "Result after hysteresis"],
    1,
    2,
)


# ##### **Wrap up**
# Once you have seen how the Canny edge detector works step by step, implement the following `canny_edge_detector` function that wraps all the previous steps and returns the final image.

# In[59]:


def canny_edge_detector(
    img: np.ndarray, sigma: int = 1,
    low_threshold: int = 50, high_threshold: int = 100
) -> np.ndarray:
    """
    This function implements the Canny edge detector.
    Put together the previous functions to implement the
    Canny edge detector.
    Don't apply the gaussian filter if sigma=0.

    Args:
        img (np.ndarray): input image
        sigma (int): standard deviation for the gaussian filter;
            if sigma=0, no gaussian filter is applied
        low_threshold (int): low threshold
        high_threshold (int): high threshold

    Returns:
        out_img (np.ndarray): image after applying the Canny edge
            detector
    """
    weak_value = 100

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    return out_img


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[61]:


img = rgb2gray(np.array(Image.open("images/pinwheel.jpg")))
out_img = canny_edge_detector(img, sigma=2, low_threshold=35, high_threshold=100)

plot_pictures(
    [img, out_img],
    ["Input image", "Canny edge detector output"],
    1,
    2,
)


# #### **2.4.1: Template matching**

# In this section, you are going to implement a template matching algorithm.
# The goal of template matching is to find the patch in the image that is most similar to a given template.
# In this exercise, you are going to use the Canny edge detector to extract the edges of the image and the template.
# Then, you will find the patch in the image that is most similar to the template.
# To choose the patch, you are asked to implement the [Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index) metric:
# \begin{equation}
# \mathrm{Jaccard}(I_{ij}, T)
# = \frac{|I_{ij} \cap T|}{|I_{ij} \cup T|}
# %= \frac{\verb|sum|(I_{ij} \land T)}{\verb|sum|(I_{ij} \lor T)}
# \end{equation}
# 
# In practice we can compute this by computing the ratio between the number of pixels where both $T$ and $I_{ij}$ have an edge and the number of pixels where $T$ or $I_{ij}$ have an edge.
# Note that $T$ denotes the template image, $I$ the target image, $I_{ij}$ a patch of $I$ at position $ij$ having the same dimensions of $T$.

# Implement the following function that takes as input the target image $I$, and the template image $T$. The returned values are a matrix of shape `(i_height-t_heigh, i_width-t_width)` containing the metric scores at each position and the indexes $(i,j)$ of the patch in $I$ with the best metric value, as depicted below. Note that $i$ indexes along the height of $I$, $j$ indexes along the width of $I$.
# 
# ```
# (i,j)-------width--------+
#   |                      |
#   |                      |
# height                   |
#   |                      |
#   |                      |
#   +----------------------+
# ```

# In[62]:


def jaccard(patch:np.ndarray,template:np.ndarray) -> float:
    """
    This function computes the Jaccard similarity between
    two images.

    Args:
        patch (np.array): patch of the input image
        template (np.array): template to match

    Returns:
        jaccard_score (float)
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    return jaccard_score

def match_template(
    img: np.ndarray, template: np.ndarray
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    This function implements the template matching algorithm.

    Args:
        img (np.ndarray): input image.
        template (np.array): template to match.

    Returns:
        img_score (np.ndarray): metric scores after applying the template matching algorithm.
        best_match_idxs (tuple): index of the best match.
    """
    t_height, t_width = template.shape
    i_height, i_width = img.shape
    assert (i_height >= t_height) and (
        i_width >= t_width
    ), f"Cannot match template of shape {template.shape} with image of shape {img.shape}"
    img_score = np.zeros((i_height - t_height + 1, i_width - t_width + 1))

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    return img_score, best_match_idxs


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# *Note: by running the following code, you should be able to detect the bigger Wally.*

# In[64]:


import matplotlib.patches as patches

img_1 = rgb2gray(np.array(Image.open("images/wally.jpg")))
img_2 = rgb2gray(np.array(Image.open("images/findwally.jpg")))

template = canny_edge_detector(img_1, sigma=1.5, low_threshold=40, high_threshold=100)
target = canny_edge_detector(img_2, sigma=1, low_threshold=25, high_threshold=100)

imgs = [img_1, np.array(Image.open("images/findwally.jpg")), img_2]
xlabels = ["Template", "Target", "Matched"]
axs = plot_pictures(imgs, xlabels, 1, 3, show=False, vmin=None, vmax=None)
img_score, best_match = match_template(target, template)

axs[-1].plot(best_match[1], best_match[0], "ro")
axs[-1].add_patch(
    patches.Rectangle(
        (best_match[1], best_match[0]),
        template.shape[1],
        template.shape[0],
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
)

plt.show()


# Here you can look at the template and at target image side by side, after having applied the Canny edge detector.
# Notice how they are different even though the parameters of the Canny edge detector are the same: this is due to the images not having the same background.

# In[65]:


imgs = [template, target]
xlabels = ["Template edges", "Target edges"]
plot_pictures(imgs, xlabels, 1, 2, show=True, cmap="gray", same_scale=True)


# ### **2.3 Harris Corner Detector** ( points)
# 
# The Harris Corner Detector is a feature detection method used in computer vision to find corners, which are points with significant intensity changes in multiple directions. It works by analyzing image gradients to identify regions where shifting a small window results in large intensity variations. 
# 
# It can be considered a variation of the Canny Edge Detector. Implement it by leveraging the previously implemented functions.
# 
# For more information: [link](https://en.wikipedia.org/wiki/Harris_corner_detector)

# In[66]:


def harris_corner_detector(
    img: np.ndarray, 
    sigma: int = 1, 
    k: float = 0.04, 
    threshold: float = 1e-2) -> np.ndarray:
    """
    Harris Corner Detector.

    Args:
        img (np.ndarray): Input grayscale image.
        sigma (int): Standard deviation for Gaussian smoothing.
        k (float): Sensitivity factor to separate corners from edges.
        threshold (float): Threshold for detecting corners.

    Returns:
        corners (np.ndarray): Image with corners marked.
    """
    #####################################################
    ##              COMPLETE THIS CODE                 ##
    #####################################################
    # Step 1: Smooth the image
    img = _ if sigma != 0 else img

    # Step 2: Compute the gradients
    dx, dy = _, _
    dx2 = dx**2
    dxy = dx*dy
    dy2 = dy**2

    # Step 3: Compute the sums of the products of gradients
    window = np.ones((3, 3))
    Sxx = convolve2d(_, window, mode='same')
    Sxy = convolve2d(_, window, mode='same')
    Syy = convolve2d(_, window, mode='same')

    # Step 4: Compute the determinant and trace of the Harris matrix
    det = _
    trace = _

    # Step 5: Compute the Harris response
    R = _

    corners = np.zeros_like(img)
    corners[_] = 255
    return corners


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[68]:


img = rgb2gray(np.array(Image.open("images/pinwheel.jpg")))
imgs = [img, harris_corner_detector(img, 1)]
labels = ["Input Image", "Harris Corner Detector Image"]
plot_pictures(imgs, labels, nrows=1, ncols=2, cmap="grey", vmin=0, vmax=255)


# ### **2.4 Gaussian Pyramid** *(2 points)*
# 
# Now let's implement a multi-scale object detection algorithm with Gaussian pyramid and template matching.

# **Gaussian Pyramid**
# 
# Implement a downscaling function that makes use of the iterative approach described by the Gaussian Pyramid. Use [`skimage.transform.resize`](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize) to resize the image to a lower scale, but don't forget to first smooth the image.
# Repeat this process `step` times.

# In[69]:


def downscale(
    img: np.ndarray,
    factor: float,
    steps:int=3,
    sigma:int=1
    ) -> np.ndarray:
    """
    Iteratively apply a gaussian blur to avoid aliasing
    and then downscale the input image by the given factor.

    Args:
        img (np.ndarray): Input image.
        factor (float): Resizing factor, how may times smaller
            is the image going to be.
        steps(int): the number of times to repeat the iteration.
        sigma (float): Standard deviation for the Gaussian filter.

    Returns:
        imgs (list): list of downscaled images. Each one is
            "factor" times smaller then the previous.
        blurred (list): list of blurred images produced in
            the downscaling process.
    """
    imgs = [img]
    blurred = []
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    return imgs, blurred


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[71]:


img = rgb2gray(np.array(Image.open("images/coffee.jpg")))
STEPS = 3
imgs, img_blur = downscale(img, 2, steps=STEPS, sigma=3)

for i in range(STEPS+1):
    fig, axes = plt.subplots(1, 2, figsize=(int(12* .8**(i+1)),int(8* .8**(i+1))))
    plt.gray()
    fig.tight_layout()
    axes[0].imshow(imgs[i])
    axes[0].set_title(f"Resized image {imgs[i].shape}" if i != 0 else "Original Image")
    axes[0].axis('off')
    if i < STEPS:
        axes[1].imshow(img_blur[i])
        axes[1].set_title(f"Smooth image {img_blur[i].shape}")
    axes[1].axis('off')


# ##### **Aliasing**
# 
# When an image is downscaled, some information is inevitably lost. If this process isn't handled carefully, it can lead to distortions such as jagged edges, moiré patterns, or a "stair-step" effect on diagonal lines. This kind of effect is called "Aliasing".
# 
# ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2F1.bp.blogspot.com%2F-AWQvtxVtBBY%2FV9F03wJwAnI%2FAAAAAAAAA7A%2FVfGtkgT8O60Y9jBx5hXT1ixw4URaqxHWQCLcB%2Fs1600%2FSupersampling%252B2D%252Banti-aliasing.gif&f=1&nofb=1&ipt=0de5f53f14ec430997ea1d210e4f6ab6bcb5131861fe84ddeeb14e15be689527&ipo=images)
# 
# These artifacts arise because the sampling rate of the new, smaller image is insufficient to accurately represent the high frequency details present in the original image. This is related to the Nyquist-Shannon sampling theorem, which states that to accurately represent a signal, the sampling rate must be at least twice the highest frequency present in the signal.
# To mitigate aliasing when downscaling, various techniques can be employed, such as applying low-pass filters before reducing the image size (i.e. a Gaussian blur).
# 
# Now confront the results of downscaling with the gaussian pyramid that you just implemented against a naive resizing.
# 
# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[72]:


img = np.array(Image.open("images/bricks.jpg"))
img_no_blur = np.array(Image.open("images/bricks.jpg"))

imgs, img_blur = downscale(img, 2, steps=STEPS, sigma=1)


for i in range(3):
    img = imgs[i+1]
    img_no_blur = resize(img_no_blur, (img_no_blur.shape[0] //2, img_no_blur.shape[1] // 2))

    fig, axes = plt.subplots(1, 2, figsize=(int(12* .75**(i+1)),int(8* .8**(i+1))))
    plt.gray()
    fig.tight_layout()
    axes[0].imshow(img.astype(int))
    axes[0].set_title(f"Downsampled image {img.shape[:2]}\nwith Anti-aliasing")
    axes[0].axis('off')
    axes[1].imshow(img_no_blur)
    axes[1].set_title(f"Downsampled image {img_no_blur.shape[:2]}\nwithout blurring")
    axes[1].axis('off')
    plt.show()


# **QUESTION**
# 
# Which differences do you notice between the first and the second image?
# What is the cause of the Aliasing effect?
# 
# -------------------------------------------------------
# 
# 
# **WRITE YOU ANSWER HERE**
# 
# 
# -------------------------------------------------------

# ...

# ### **2.5: Multi-Scale Template Matching (? Points)**
# Write a multi-scale object detection algorithm using the Gaussian Pyramid and the template matching algorithm you have implemented in the previous exercise.

# In[73]:


def multi_scale_match_template(
    img: np.ndarray,
    template: np.ndarray,
    steps: int,
    sigma: float = 1,
    low_threshold: int = 30,
    high_threshold: int = 70,
) -> List[Dict[str, Union[np.ndarray, Tuple[int, int]]]]:
    """
    This function implements the multi-scale template matching.

    Args:
        img (np.ndarray): input image.
        template (np.ndarray): template image.
        steps (int): number of steps for the Pyramid.
        sigma (float): the Gaussian filter parameter.
        low_threshold (int): low threshold for Canny edge detector.
        high_threshold (int): high threshold for Canny edge detector.

    Returns:
        out: list of dictionaries containing:
            img_score (np.ndarray): score image.
            result (Tuple[int, int]): best match coordinates.
            template (np.ndarray): template image.
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    out = [
        {"img_score": score, "result": result, "template": t}
        for (score, result), t in zip(score_results, templates)
    ]
    return out


# _____________________________________________________
# **Do not write below this line just run it**
# _____________________________________________________
# 
# 

# In[75]:


img1 = rgb2gray(np.array(Image.open("images/findwally.jpg")))
img2 = rgb2gray(np.array(Image.open("images/wally.jpg")))

multiscale_res = multi_scale_match_template(
    img=img1,
    template=img2,
    steps=2,
    sigma=1,
    low_threshold=30,
    high_threshold=70,
)


imgs = [img2, np.array(Image.open("images/findwally.jpg"))]
xlabels = [
    "Template",
    "Matched",
]
axs = plot_pictures(imgs, xlabels, 1, 2, show=False, vmin=None, vmax=None)

for d in multiscale_res:
    axs[-1].plot(d["result"][1], d["result"][0], "ro")
    axs[-1].add_patch(
        patches.Rectangle(
            (d["result"][1], d["result"][0]),
            d["template"].shape[1]-2,
            d["template"].shape[0]-2,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
    )
plt.show()


# ## **Question 3: Object Identification** *(12 Points)*

# *Note: This identification part contains **query and model images** for the evaluation, which correspond to the same set of objects photographed from different viewpoints. The files **model.txt** and **query.txt** contain lists of image files arranged so that i-th model image depicts the same object as i-th query image.*

# #### **3.1: 3D Joint Color Histogram** *(2 points)*
# 
# Your task is to implement a Python function, histogramdd, from scratch. This function should compute the multidimensional histogram of a given image. The function should accept an image and a specification for the number of bins to use for each dimension. You should then divide the data into bins and count the number of data points that fall into each bin.
# 
# If you want to check that your result is correct, you can compare it with the [function already implemented by NumPy](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html).

# In[76]:


def histogramdd(img: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Follow these steps:
        1) Create bin intervals for each dimension.
        2) Compute the shape of the histogram based on bin intervals.
        3) Initialize the histogram as an array of zeros.
        4) Compute the histogram for the input data.

    Args:
        img (np.ndarray): Input image.
        bins (int): Number of bins.

    Returns:
        histograms (np.ndarray): Histogram.
        bin_edges (np.ndarray): Bin edges.
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################


    return histograms, bin_edges


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[78]:


def histogram3dplot(h, e, fig=None):
    """
    Visualize a 3D histogram

    Parameters
    ----------

    h: histogram array of shape (M,N,O)
    e: list of bin edge arrays (for R, G and B)
    """
    M, N, O = h.shape  # numero di bins
    idxR = np.arange(M)
    idxG = np.arange(N)
    idxB = np.arange(O)

    R, G, B = np.meshgrid(idxR, idxG, idxB)  # divisione per bins
    a = np.diff(e[0])[0]
    b = a / 2
    R = a * R + b

    a = np.diff(e[1])[0]
    b = a / 2
    G = a * G + b

    a = np.diff(e[2])[0]
    b = a / 2
    B = a * B + b

    colors = np.vstack((R.flatten(), G.flatten(), B.flatten())).T / 255
    h = h / np.sum(h)
    if fig is not None:
        f = plt.figure(fig)
    else:
        f = plt.gcf()
    ax = f.add_subplot(111, projection="3d")
    mxbins = np.array([M, N, O]).max()
    ax.scatter(
        R.flatten(),
        G.flatten(),
        B.flatten(),
        s=h.flatten() * (256 / mxbins) ** 3 / 2,
        c=colors,
    )

    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")


img = Image.open("images/colors.jpg")
img = np.array(img)
h, e = histogramdd(img.reshape(-1, 3), bins=10)
histogram3dplot(h, e)


# #### **3.2: Types of Histograms** *(2 points)*
# 
# In this section, you are asked to implement the **GB** histogram and to correct the dx/dy histogram. Finally, you are asked to comment and compare the results.

# ##### **GB Histogram**

# In this exercise, you should implement the **GB** histogram.
# 
# *Note: as before, when considering GB, you should keep in mind that the range of the pixel's intensity is between $0$ and $255$*.

# In[79]:


def gb_hist(img_color_double: np.ndarray, num_bins: int = 5) -> np.ndarray:
    """
    Compute the *joint* histogram for the G and B color channels in the image.
    The histogram should be normalized so that sum of all values equals 1,
    assume that values in each channel vary between 0 and 255

    Args:
        img_color_double (np.ndarray): Input color image.
        num_bins (int): Number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2.

    Returns:
        hists (np.ndarray): Joint histogram.

    E.g. hists[0,9] contains the number of image_color pixels such that:
        - their G values fall in bin 0
        - their B values fall in bin 9
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    return hists


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[81]:


# Compose and test GB histograms (histogram_module.GB_hist)
img_color = np.array(Image.open("images/superman.jpg"))

plt.figure(3, figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_color)

num_bins_color = 5
plt.subplot(1, 2, 2)
hist_gb = gb_hist(img_color.astype("double"), num_bins_color)
plt.bar(np.array(range(1, hist_gb.size + 1)), hist_gb)
plt.show()


# ##### **DX/DY Histogram**

# In this exercise, you will find an implemented version of another type of histogram, the dx/dy one. However, **it is not correct, there are some mistakes in the code**:
# 
# - function `gaussdx`: **2 mistakes**;
# - function `gauss_dxdy`: **2 mistakes**;
# - function `hist_dxdy`: **2 mistakes**.
# 
# You are asked to copy each function in the corresponding cell and to correct the mistakes. Put a comment on the lines where you have made the corrections.
# Before coding, write down the first derivative of the Gaussian function.
# 
# *HINT: leverage the result of the derivation to correct the mistakes in `gaussdx`*.
# 
# *Note: each mistaken line counts as one mistake; a mistake can also be a missing line or a missing block of lines*.

# -------------------------------------------------------
# 
# 
# **WRITE YOU ANSWER HERE**
# 
# 
# -------------------------------------------------------

# $$
# \begin{align}
# \frac{d}{dx}g(x) &= \frac{d}{dx}\left[\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{x^2}{2\sigma^2}\right)\right] = \\
# &= \text{YOUR ANSWER HERE}
# \end{align}
# $$

# In[82]:


def gaussdx(sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function computes the first derivative of the 1D Gaussian operator.

    Args:
        sigma (float) : the standard deviation of the Gaussian filter

    Returns:
        Dx (np.ndarray): the first derivative of the 1D Gaussian operator
        x (np.ndarray): the indexes of the 1D Gaussian operator
    """
    sigma = math.ceil(sigma)
    filter_size = 3 * sigma + 1

    # Generate the index x
    zero_pos = 3 * filter_size  # the (zero_pos+1)th element is the 0 for the index
    x = np.arange(filter_size) - zero_pos  # indexes from -3*sigma to 3*sigma

    # Compute the Gaussian curve with std-dev sigma at the indexes x
    Dx = x * np.exp(-(x**2) / (2.0 * sigma**2)) / (math.sqrt(2.0 * np.pi) * sigma)

    return Dx, x


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# To check that your implementation is correct, plot the result of the function with $\sigma=3$ and compare it with the image below.
# 
# ![](./images/1gaussdx.png)

# In[84]:


# plot the first derivative of the 1D Gaussian operator
sigma = 3
Dx, x = gaussdx(sigma)
plt.figure(1)
plt.plot(x, Dx, ".-")
plt.show()


# _____________________________________________________

# In[88]:


def gauss_dxdy(img: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function applies the first derivative of the 1D Gaussian operator to the image in the x and y directions.

    Args:
        img (np.ndarray): the input image.
        sigma (float): the standard deviation of the Gaussian filter.

    Returns:
        img_Dx (np.ndarray): the image after applying the first derivative of the 1D Gaussian operator in the x direction.
        img_Dy (np.ndarray): the image after applying the first derivative of the 1D Gaussian operator in the y direction.
    """
    #####################################################
    ##                 FIX THIS CODE                   ##
    #####################################################
    Gx, _ = gauss(sigma, kappa=3)
    Dx, _ = gaussdx(sigma)

    img_Dx = convolve1d(convolve1d(img, Dx), Gx)
    img_Dy = convolve1d(convolve1d(img, Gx), Dx)

    return img_Dx, img_Dy


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# To check that your implementation is correct, plot the result of the function with $\sigma=1$ and compare it with the images below.
# 
# ![](./images/check_gaussdxdy.png)

# In[94]:


img_color = rgb2gray(np.array(Image.open("images/pinwheel.jpg")))

img_Dx, img_Dy = gauss_dxdy(img_color, 1)

plot_pictures([img_Dx, img_Dy], ["Dx", "Dy"], 1, 2)


# --------------------------------------------

# In[95]:


def hist_dxdy(img_gray, num_bins=5):
    """
    This function computes the *joint* histogram of Gaussian partial derivatives of the image in x and y direction.
    Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6].
    The histogram should be normalized so that the sum of all values equals 1.

    Args:
        img_gray: the input image
        num_bins: number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2

    Returns:
        hists: the joint normalized histogram of Gaussian partial derivatives of the image in x and y direction
    """

    assert len(img_gray.shape) == 2, "image dimension mismatch"
    assert img_gray.dtype == "float", "incorrect image type"
    
    #####################################################
    ##                 FIX THIS CODE                   ##
    #####################################################

    # Compute the first derivatives of img_gray
    sigma = 3.0
    img_dx, img_dy = gauss_dxdy(img_gray, sigma)

    # Set the min_der and max_der to -6 and 6, which defines the ranges for quantization
    min_der, max_der = (-6, 6)

    # Flatten the 2D derivative images to 1D arrays
    img_dx = img_dx.reshape(-1)
    img_dy = img_dy.reshape(-1)

    # Clip the min and max values to min_der and max_der respectively
    # and shift minumum values to 0
    img_dx = np.clip(img_dx, min_der, max_der) + max_der
    img_dy = np.clip(img_dy, min_der, max_der)

    hists = np.zeros((num_bins, num_bins), dtype=int)
    bin_range = (max_der - min_der) / num_bins

    # quantize image derivative valuer into bins
    bin_dx = np.floor(img_dx / bin_range).astype(int)
    bin_dy = np.floor(img_dy / bin_range).astype(int)
    bin_dx = np.clip(bin_dx, 0, num_bins - 1)
    bin_dy = np.clip(bin_dy, 0, num_bins - 1)

    for i in range(bin_dx.size):
        hists[bin_dx[i], bin_dy[i]] += 1

    hists = hists.flatten()
    return hists


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# To check that your implementation is correct, compare it with the image below.
# 
# ![](./images/check_hist_dxdy.png)

# In[97]:


img_color = np.array(Image.open("images/pinwheel.jpg"))
img_gray = rgb2gray(img_color.astype("double"))

plt.figure(5)
plt.subplot(1, 2, 1)
plt.imshow(img_color)

num_bins_dxdy = 10
plt.subplot(1, 2, 2)
hist_dxdy_ = hist_dxdy(img_gray, num_bins_dxdy)
plt.bar(np.array(range(1, hist_dxdy_.size + 1)), hist_dxdy_)
plt.show()


# ##### **Theoretical questions**

# Now that you have implemented two color histograms - 3D joint and GB - and the dx/dy histogram, answer to the following questions:
# 
# 1. What does the dx/dy histogram represent in terms of image features?
# 2. In your opinion, which of the two types of histograms (color and dx/dy) is more robust to changes in illumination? Why?

# -------------------------------------------------------
# 
# 
# **WRITE YOU ANSWER HERE**
# 
# 
# -------------------------------------------------------

# ### **3.3: Histogram Metrics** *(2 points)*
# 
# Now that you have implemented and used different types of histograms, let's consider distance functions.
# 1. Write the definition of intersection distance function, $L_2$, and $\chi^2$ distance functions.
# 2. Implement each one of them.
# 3. Write down some considerations on each one of them.

# Complete the definition of each metric.
# 
# * Intersection function **for unnormalized histograms**:
# 
# \begin{equation}
# \bigcap(Q,V)=
# \end{equation}
# 
# * $L_2$ function:
# \begin{equation}
# d(Q,V)=
# \end{equation}
# 
# * $\chi^2$ function:
# \begin{equation}
# \chi^2(Q,V)=
# \end{equation}
# 
# Unusual metric:
# 
# * Negative Cross-Entropy function:
# \begin{equation}
# H(P, Q) = 
# \end{equation}
# 
# 
# 

# In[98]:


def hist_intersect(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the intersection between histograms x and y.
    Check that the distance range is [0,1].

    Args:
        h1 (np.ndarray): Input histogram.
        h2 (np.ndarray): Input histogram.

    Returns:
        x (float): Intersection distance between histograms x and y.
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################


    return x


def hist_l2(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the L2 between x and y histograms.
    Check that the distance range in [0,sqrt(2)].

    Args:
        h1 (np.ndarray): Input histogram.
        h2 (np.ndarray): Input histogram.

    Returns:
        x (float): L2 distance between x and y histograms.
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    return x


def hist_chi2(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute chi2 between x and y.
    Check that the distance range in [0,Inf].

    Args:
        h1 (np.ndarray): Input histogram.
        h2 (np.ndarray): Input histogram.

    Returns:
        x (float): Chi2 distance between x and y.
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    return x


def hist_ce(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Compute the cross-entropy between two histograms.

    Args:
        h1 (np.ndarray): First input histogram.
        h2 (np.ndarray): Second input histogram.

    Returns:
        float: Cross-entropy between h1 and h2.
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    return ce


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[100]:


img1 = np.array(Image.open("model/obj100__0.png"))
img2 = np.array(Image.open("model/obj63__0.png"))

num_bins_color = 5
hist1 = gb_hist(img1.astype("double"), num_bins_color)
hist2 = gb_hist(img2.astype("double"), num_bins_color)

# plot images and histograms
plt.figure(6, figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.show()

plt.figure(7, figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.bar(np.array(range(1, hist1.size + 1)), hist1)
plt.subplot(1, 2, 2)
plt.bar(np.array(range(1, hist2.size + 1)), hist2)
plt.show()

metrics = np.array(
    [hist_intersect(hist1, hist2), hist_l2(hist1, hist2), hist_chi2(hist1, hist2), hist_ce(hist1, hist2)]
)

# write distances to file
np.savetxt("assets/metrics.npy", metrics)

# read metrics from file
metrics = np.loadtxt("assets/metrics.npy")

# print distances and compare them with the solution
print(f"Intersection:\n\t{metrics[0]}\t{hist_intersect(hist1, hist2)}")
print(f"L2:\n\t{metrics[1]}\t{hist_l2(hist1, hist2)}")
print(f"Chi2:\n\t{metrics[2]}\t{hist_chi2(hist1, hist2)}")
print(f"CE:\n\t{metrics[3]}\t{hist_ce(hist1, hist2)}")


# ### **3.4: Image Retrieval** *(2 points)*
# 
# Now that you have implemented and used different types of histograms, it's time to test how suitable they are for retrieving images in a query-by-example scenario.
# 
# Implement a function called `find_best_match` that returns the closest model images for each query image.
# The function takes input string parameters, identifying the distance function, histogram function, and the number of histogram bins.
# 
# *Note: See comments at the beginning of the `find_best_match` function for more details.*

# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[101]:


def is_grayvalue_hist(hist_name: str) -> bool:
    """
    Handle function to discriminate when your input
    function is in gray_scale or colors.

    Args:
        hist_name (str): histogram name.

    Returns:
        bool: True if the histogram is in gray_scale, False otherwise.
    """
    if hist_name == "grayvalue" or hist_name == "dxdy":
        return True
    elif hist_name == "rgb" or hist_name == "gb":
        return False
    else:
        assert False, "unknown histogram type"


def get_hist_by_name(img: np.ndarray, num_bins_gray: int, hist_name: str) -> np.ndarray:
    """
    Handle function to get the correct historgam function
    by his name.

    Args:
        img (np.ndarray): input image.
        num_bins_gray (int): number of bins for the gray_scale histogram.

    Returns:
        np.ndarray: histogram.
    """
    # if hist_name == "rgb":
    #     return rgb_hist(img, num_bins_gray)
    if hist_name == "gb":
        return gb_hist(img, num_bins_gray)
    elif hist_name == "dxdy":
        return hist_dxdy(img, num_bins_gray)
    else:
        assert False, "unknown hist type: %s" % hist_name


def get_dist_by_name(x: np.ndarray, y: np.ndarray, dist_name: str) -> float:
    """
    Handle function to get the correct distance function
    by his name.

    Args:
        x (np.ndarray): input histogram.
        y (np.ndarray): input histogram.

    Returns:
        float: distance.
    """
    if dist_name == "chi2":
        return hist_chi2(x, y)
    elif dist_name == "intersect":
        return 1 - hist_intersect(x, y)
    elif dist_name == "l2":
        return hist_l2(x, y)
    elif dist_name == "ce":
        return hist_ce(x, y)
    elif dist_name == "all":
        pass
    else:
        assert False, "unknown distance: %s" % dist_name


def read_files(
    model_path: str = "assets/model.txt", query_path: str = "assets/query.txt"
) -> Tuple[List[str], List[str]]:
    """
    Handle function to read query and model files.

    Args:
        model_path (str): path to the model file.
        query_path (str): path to the query file.

    Returns:
        Tuple[List[str], List[str]]: query images and model images.
    """
    with open(model_path) as fp:
        model_images = fp.readlines()
    model_images = [x.strip() for x in model_images]

    with open(query_path) as fp:
        query_images = fp.readlines()
    query_images = [x.strip() for x in query_images]

    return query_images, model_images


# In[102]:


# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain
#       handles to distance and histogram functions, and to find out whether histogram function
#       expects grayvalue or color image


def compute_histograms(
    image_list: List[np.ndarray], hist_type: str, hist_isgray: bool, num_bins: int
) -> List[np.ndarray]:
    """
    Compute the histogram for each image in image_list.

    Args:
        image_list (List[np.ndarray]): list of images.
        hist_type (str): histogram type.
        hist_isgray (bool): True if the histogram is in gray_scale, False otherwise.
        num_bins (int): number of bins for the gray_scale histogram.

    Returns:
        image_hist (List[np.ndarray]): list of histograms.
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################



    return image_hist

def find_best_match(
    model_images: List[str],
    query_images: List[str],
    dist_type: str,
    hist_type: str,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to find the best match for each image in the
    query folder.

    Args:
        model_images (List[str]): list of strings with the path of model images.
        query_images (List[str]): list of strings with the path of query images.
        dist_type (str): a string to represent the name of the distance you want to use. Should be one among "l2", "intersect", "chi2".
        hist_type (str): a string to represent the name of the histogram you want to use. Should be one among "grayvalue", "rgb", "rg", "dxdy".
        num_bins (int): number of bins for the gray_scale histogram.

    Returns:
        best_match (np.ndarray): list containing in each position the index of the retrieved best matching image.
        D (np.ndarray): Matrix with |model_images| rows and |query_images| columns containing the scores of each matching.
    #"""
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################


    return best_match, D


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[104]:


# model_images - list of file names of model images
# query_images - list of file names of query images

query_images, model_images = read_files()

dist_type = "intersect"
hist_type = "gb"
num_bins = 10

[best_match, D] = find_best_match(
    model_images, query_images, dist_type, hist_type, num_bins
)

print(
    "Indexes of correct retrieved images is:\n ",
    *np.where(best_match == np.arange(len(query_images)))
)
print(
    "The Recognition rate is",
    sum(best_match == np.arange(len(query_images))) / len(query_images),
)


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# ### **Report** *(2 points)*
# 
# Experiment with different functions and numbers of histogram bins, and find a combination that works best. **Submit the summary of your experiments in a report as part of your solution.**
# 

# ----------------------------
# 
# **Fill the table below**
# 
# ----------------------------
# 
# 
# |  | Number of Bins | Metric(Distance Metric) | Accuracy |
# |---|---|---|---|
# | 1 |  |  |  |
# | 2 |  |  |  |
# | 3 |  |  |  |
# | ... |  |  |  |
# 

# ## **Question 4: Performance Evaluation** *(5 Points)*

# In this question, your goal is to implement different types of metrics to evaluate the performance of a binary classificator.
# 
# For the purpose of this exercise you don't need to know the inner workings of the ML model, therefore imagine to have a black box model that estimates the probability of an event occurring, such as the sky being clear or not, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1.
# 
# In order to compute a performance score (e.g., accuracy), you need to compare your predictions with the true values. So you need to convert the predicted probabilities into 0s and 1s by means of a threshold.
# 
# What you have to do is the following:
# - **compute the performance** of the model with all the possible tresholds in the interval \[0.0, 1.0\] with increments of 0.5
# - **analyze** how the metrics vary at different thresholds and **write a report** on your analysis

# ### **4.1: Performance Evaluation** *(2 points)*
# 
# Write the code to compute the metrics for performance evaluation, then run the provided code cell to check if your results are correct.

# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[105]:


data = pd.read_csv("./assets/ionosphere.data", header=None)
X = data[data.columns[:34]]
y = data[data.columns[-1]]
y = np.array([0 if value == "b" else 1 for value in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]


# In[106]:


def my_accuracy_score(y_true: list, y_pred: list) -> float:
    """
    Compute the accuracy score.

    Args:
        y_true (list): list of true labels.
        y_pred (list): list of predicted labels.

    Returns:
        float: accuracy score.
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################

    return accuracy


def my_precision_score(y_true: list, y_pred: list) -> float:
    """
    Compute the precision score.

    Args:
        y_true (list): list of true labels.
        y_pred (list): list of predicted labels.

    Returns:
        float: precision score.
    """  #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    return precision


def my_recall_score(y_true: list, y_pred: list) -> float:
    """
    Compute the recall score.

    Args:
        y_true (list): list of true labels.
        y_pred (list): list of predicted labels.

    Returns:
        float: recall score.
    """  #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    return recall


def my_f1_score(y_true: list, y_pred: list) -> float:
    """
    Compute the f1 score.

    Args:
        y_true (list): list of true labels.
        y_pred (list): list of predicted labels.

    Returns:
        float: f1 score.
    """  #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    return f1


def my_confusion_matrix(y_true: list, y_pred: list) -> np.ndarray:
    """
    Compute the confusion matrix.

    Args:
        y_true (list): list of true labels.
        y_pred (list): list of predicted labels.

    Returns:
        np.ndarray: confusion matrix.
    """  #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    return np.array(matrix, dtype=np.int64)


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[108]:


# CHECK IF YOUR METRICS ARE CORRECT BY RUNNING THIS CODE CELL


def compare_metric(my_metric_fun, metric_fun, y_test, preds):
    try:
        if my_metric_fun.__name__ != "my_confusion_matrix":
            assert round(my_metric_fun(y_test, preds), 4) == round(
                metric_fun(y_test, preds), 4
            )
        else:
            assert np.array_equal(
                my_metric_fun(y_test, preds), metric_fun(y_test, preds)
            )
        print(f"{my_metric_fun.__name__} is correct")
    except:
        print(f"{my_metric_fun.__name__} is wrong")


preds = np.array([0 if p < 0.5 else 1 for p in y_pred])
compare_metric(my_accuracy_score, accuracy_score, y_test, preds)
compare_metric(my_precision_score, precision_score, y_test, preds)
compare_metric(my_recall_score, recall_score, y_test, preds)
compare_metric(my_f1_score, f1_score, y_test, preds)
compare_metric(my_confusion_matrix, confusion_matrix, y_test, preds)


# ### **4.2: Closest Neighbours** *(2 points)*
# 
# Implement a function `show_neighbors`, which inputs a list of model images and a list of query images.
# For each query image, the function has to output a visualization of 5 model images which are most similar to the query image according to the specified distance metric.
# 
# *HINT: use the function `find_best_match`.*
# 
# ![](images/neighbors.png)
# 

# In[109]:


def show_neighbors(
    model_images: List[str],
    query_images: List[str],
    dist_type: str,
    hist_type: str,
    num_bins: int,
) -> None:
    """
    For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.

    Note: use the previously implemented function 'find_best_match'

    Args:
        model_images (List[str]): list of strings with the path of model images.
        query_images (List[str]): list of strings with the path of query images.
        dist_type (str): a string to represent the name of the distance you want to use. Should be one among "l2", "intersect", "chi2".
        hist_type (str): a string to represent the name of the histogram you want to use. Should be one among "grayvalue", "rgb", "rg", "dxdy".
        num_bins (int): number of bins for the gray_scale histogram.

    Returns:
        None
    """

    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################


# In[111]:


## visualize nearest neighbors
query_images_vis = [query_images[i] for i in np.array([0, 4, 9])]
show_neighbors(model_images, query_images_vis, 'intersect', hist_type, num_bins)


# ### **4.3: Retrieval Metrics (? points)**
# In previous sections, we implemented classical metrics for classification tasks. However, since we've also created a find_best_match function, we can approach this as a retrieval task. This perspective allows us to find the top-k items that best match the input item.
# Given this retrieval approach, it's more appropriate to use a metric specifically designed for retrieval tasks.
# 
# **Top-k Accuracy**
# 
# This is a variation of the classical accuracy metric. Here's how it works:
# 1. For each input item, we consider the top k results.
# 2. We count the result as positive if at least one of these k results matches the correct label.

# In[112]:


def topk_accuracy(y_true, y_pred, k:int) -> float:
    """
    Compute the Top-K accuracy, i.e. check whether the
    correct label is in the top-k results.

    Args:
        y_true (list): list of correct labels
        y_pred (np.array): A matrix of size (num_samples, num_results)
            where each cell contains the predicted score for each
            element in the dataset
        k (int): number of best matching items to consider

    Returns:
        topk_accuracy (float)
    """
    #####################################################
    ##                 YOUR CODE HERE                  ##
    #####################################################
    return result


# --------------------------------------------
# **Do not write below this line just run it**
# 
# --------------------------------------------

# In[114]:


if 'D' not in globals():
    _,D = find_best_match(
        model_images, query_images, dist_type, hist_type, num_bins
    )
y_true = np.arange(D.shape[0])

topk_accuracy(y_true, D, k=5)


# Now that you understand how the Top-$k$ accuracy works:
# - Compute the top-k accuracy for each of the values of $k \in \{1,...,20\}$.
# - Produce a plot, with the value $k$ on the x-axis and the topk-accuracy on the y-axis.
# - Which is the minimum value of $k$ for which the top-$k$ accuracy is at least 80%?
# - Which are the tradeoffs of incrementing $k$?
# 

# In[115]:


K = 20
y_true; # These are the correct labels

#####################################################
##                 YOUR CODE HERE                  ##
#####################################################


# -------------------------------------------------------
# 
# 
# **WRITE YOU ANSWER HERE**
# 
# 
# -------------------------------------------------------

# ...

# ### **4.4: Analysis and Report** *(1 points)*
# 
# 
# Consider again the metrics for the classifier in Question 4.1.
# Write the code to compute the metrics for all the possible tresholds in the interval (0.0, 1.0) with increments of 0.5 of `y_pred`.  
# Ideally, you should put everything into a pandas DataFrame and print it for a better visualization.  
# Then, write a report analyzing you your results. You have complete freedom on your analysis.
# 
# Imagine also to be in a medical scenario where your model is predicting the presence or not of a desease and try to answer to the following questions:
# - What is the real life trade-off between high precision and high recall? What would you choose?
# - Why isn't the accuracy an appropriate measure in such scenario?

# In[117]:


# Use this cell as you'd like

