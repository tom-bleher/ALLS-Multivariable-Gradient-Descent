
### Project description
What does the code do?

- Program is ran
- Code waits for new images to appear in specified path
- Once a new Image is detected it's processed for its mean brightness
- The code continues, summing the individual mean of single images until we have `self.n_images` to take total average over the single averages sum
- The code takes an initial guess for each parameter (+1 or -1)
- The code takes the partial derivatives ($\frac{\partial C}{\partial \phi_{2}}$, $\frac{\partial C}{\partial \phi_{3}}$, $\frac{\partial C}{\partial f}$), and adds them to their respective derivative parameter history lists
- We adjust the new parameter value relying on the latest parameter derivative using gradient ascent so that $a_{n+1}=a_{n}+\gamma \frac{\partial C(\phi_{2},\phi_{3},f)}{\partial p}, \; \text{where} \; p \in P,\; \text{and} \; P={\{\phi_{2}, \phi_{3},f\}}$
- Repeat until the function $C(\phi_{2},\phi_{3},f)$ is at maximum (or at least close to it)

### Code setup
##### Camera triggering and capture
We utilize a physical pulse generator (from master clock or photodiode) for the pulse generation to synchronize the camera triggering to laser. 

- Open the SpinView program
- Connect camera to triggering Stanford box and computer
- Adjust region of interest and bit depth to `16 bit`
- Switch `trigger mode` to on
- Click on record function 
- Set format to `.tiff` and recording mode to `Streaming`
- Select directory for the code to access 

##### Install imports
The following libraries are used for the projects, ensure you have installed the necessary libraries before proceeding.

```python
import os
import cv2 # needs to be installed 
import numpy as np # needs to be installed 
from ftplib import FTP
import shutil
import random
from watchdog.events import FileSystemEventHandler # needs to be installed 
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets # needs to be installed 
from pyqtgraph.Qt import QtGui # needs to be installed 
import pyqtgraph as pg # needs to be installed 
import sys
import time
from watchdog.events import FileSystemEventHandler # needs to be installed 
from watchdog.observers import Observer # needs to be installed 
```

##### Hard-coded paths
Marked by full capitalization, the code includes a number of hard coded paths, before proceeding adjust them accordingly to your setup. 

```python
# computer near chamber
MIRROR_FILE_PATH = r'mirror_command/mirror_change.txt' # this is the txt file the code writes to for the mirror
DISPERSION_FILE_PATH = r'dazzler_command/dispersion.txt' # this is the txt file the code writes to for the dazzler
self.IMG_PATH = r'C:\Users\blehe\Desktop\Betatron\images' # this is the folder from which the code will process the images, make sure it aligns with the path specified in SpinView

self.MIRROR_HOST = "192.168.200.3" # ip of deformable mirror computer
self.MIRROR_USER = "Utilisateur" # windows user of deformable mirror computer
self.MIRROR_PASSWORD = "alls" # windows user password of deformable mirror computer

self.DAZZLER_HOST = "192.168.58.7" # ip of dazzler mirror computer
self.DAZZLER_USER = "fastlite" # windows user of dazzler computer
self.DAZZLER_PASSWORD = "fastlite" # windows user password of dazzler computer    
```

## Plotting
The code includes a number of plots to help actively track the evolution of system. 

![[plot.svg|center|800]]
- `count` vs `n_images iteration` $C(\phi_{2},\phi_{3},f)(n)$ 
- `count_focus_derivative` vs `n_images iteration` $\frac{\partial C}{\partial f}(n)$
- `count_second_dispersion_derivative` vs `n_images iteration` $\frac{\partial C}{\partial\phi_{2}}(n)$
- `count_third_dispersion_derivative` vs `n_images iteration`  $\frac{\partial C}{\partial\phi_{3}}(n)$
- `total gradient` vs `n_images iteration` ${[\frac{\partial C}{\partial f}+ \frac{\partial C}{\partial\phi_{2}}+\frac{\partial C}{\partial\phi_{3}}](n)}$

## Image processing 

##### X-ray Count
Since we're imaging a phosphor screen, the count of X-ray from the LWFA is directly proportional to the brightness of the image. The following function calculates the average brightness per pixel of an image. 

```python 
def calc_xray_count(image_path):
    original_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR)
    median_filtered_image = cv2.medianBlur(original_image, 5)
    median_filtered_image += 1  # Avoid not counting black pixels in image
    pixel_count = np.prod(median_filtered_image.shape)
    img_brightness_sum = np.sum(median_filtered_image)
    if (pixel_count > 0):
        img_avg_brightness = (img_brightness_sum/pixel_count) -1 # Subtract back to real data
    else:
        img_avg_brightness = 0
    return img_avg_brightness
```

`cv2.IMREAD_ANYDEPTH` makes sure we're reading in the bit depth of the image, `16bit`.

## Optimization algorithm
Processing the data with vanilla gradient descent to optimize the `count` function by adjusting `second_order_dispersion`, `third_order_dispersion`, and `focus` according to the real-time count reading from the camera.

```python
def process_images(self, new_images):
	self.initialize_image_files()
	new_images = [image_path for image_path in new_images if os.path.exists(image_path)]
	new_images.sort(key=os.path.getctime)
	for image_path in new_images:
		relative_path = os.path.relpath(image_path, self.IMG_PATH)
		img_mean_count = self.calc_xray_count(image_path) # calculate count per image
		self.n_images_count_sum += np.sum(img_mean_count) # add to temporary variable that includes the sum of individual mean count
		self.run_count += 1
		if self.run_count % self.n_images == 0: 
			self.mean_count_per_n_images = np.mean(img_mean_count) # we take the mean of the sum of individual mean count of single images.
			self.count_history.append(self.mean_count_per_n_images)
			self.n_images_run_count += 1 # this is a special run count for each n_images processed
			self.iteration_data.append(self.n_images_run_count)
			if self.n_images_run_count == 1: # for the initial round let's tale a random guess
				print('-------------')                    
				self.focus_history.append(self.initial_focus)                      
				self.second_dispersion_history.append(self.initial_second_dispersion)
				self.third_dispersion_history.append(self.initial_third_dispersion)
                    
				print(f"initial values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}, third_dispersion {self.third_dispersion_history[-1]}")
				print(f"initial directions are: focus {self.random_direction[0]}, second_dispersion {self.random_direction[1]}, third_dispersion {self.random_direction[2]}")
				self.initial_optimize() # take random guesses for each variable`
				
			else:
				self.n_images_dir_run_count += 1 # this is a special count that's used for the plotting of the derivatives 
				self.optimize_count() # this is the main optimization algorithm
			with open(MIRROR_FILE_PATH, 'w') as file: # write to the mirror file
				file.write(' '.join(map(str, mirror_values)))
			with open(DISPERSION_FILE_PATH, 'w') as file: # write to the dazzler file
				file.write(f'order2 = {dispersion_values[0]}\n')
				file.write(f'order3 = {dispersion_values[1]}\n')
			QtCore.QCoreApplication.processEvents()

			print(f"mean_count_per_{self.n_images}_images {self.count_history[-1]}, current values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}, third_dispersion {self.third_dispersion_history[-1]}")
			self.upload_files() # send files to computers

			# update the plots 
			self.plot_curve.setData(self.iteration_data, self.count_history)
			self.focus_curve.setData(self.der_iteration_data, self.focus_der_history)
			self.second_dispersion_curve.setData(self.der_iteration_data, self.second_dispersion_der_history)
			self.third_dispersion_curve.setData(self.der_iteration_data, self.third_dispersion_der_history)
			self.total_gradient_curve.setData(self.der_iteration_data, self.total_gradient_history)

			# reset image processing variables 
			self.n_images_count_sum = 0
			self.mean_count_per_n_images  = 0
			img_mean_count = 0  
			print('-------------')
```

##### Initial optimization
```python
def initial_optimize(self):
	# take initial guesses for each variable 
	self.new_focus = self.focus_history[-1] + self.random_direction[0]
	self.new_second_dispersion = self.second_dispersion_history[-1] + self.random_direction[1]
	self.new_third_dispersion = self.third_dispersion_history[-1] + self.random_direction[2]

	# bind to bounds and round to round to integer
	self.new_focus = round(np.clip(self.new_focus, self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND))
	self.new_second_dispersion = round(np.clip(self.new_second_dispersion, self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND))
	self.new_third_dispersion = round(np.clip(self.new_third_dispersion, self.THIRD_DISPERSION_LOWER_BOUND, self.THIRD_DISPERSION_UPPER_BOUND))

	# add to lists
	self.focus_history.append(self.new_focus)
	self.second_dispersion_history.append(self.new_second_dispersion)
	self.third_dispersion_history.append(self.new_third_dispersion)

	# update in file
	mirror_values[0] = (self.focus_history[-1])
	dispersion_values[0] = (self.second_dispersion_history[-1])
	dispersion_values[1] = (self.third_dispersion_history[-1])
```

##### Main optimization
The main optimization uses the gradient descent algorithm to adjust the variables and lead to peak count. After we've reached a sufficient point (either small change in count or repetition in all three parameters) we stop the optimization algorithm.

```python
    def calc_derivatives(self):
	    #take the partial derivatives approximation
        self.count_focus_der = (self.count_history[-1] - self.count_history[-2]) / (self.focus_history[-1] -self.focus_history[-2])
        self.count_second_dispersion_der = (self.count_history[-1] - self.count_history[-2]) / (self.second_dispersion_history[-1] - self.second_dispersion_history[-2])
        self.count_third_dispersion_der = (self.count_history[-1] - self.count_history[-2]) / (self.third_dispersion_history[-1] - self.third_dispersion_history[-2])
        self.focus_der_history.append(self.count_focus_der)
self.second_dispersion_der_history.append(self.count_second_dispersion_der)
        self.third_dispersion_der_history.append(self.count_third_dispersion_der)
        self.total_gradient = (self.focus_der_history[-1] + self.second_dispersion_der_history[-1] + self.third_dispersion_der_history[-1])
        self.total_gradient_history.append(self.total_gradient)
        self.der_iteration_data.append(self.n_images_dir_run_count)
        
        return {"focus":self.count_focus_der,"second_dispersion":self.count_second_dispersion_der,"third_dispersion":self.count_third_dispersion_der}
```

```python
    def optimize_count(self):
        derivatives = self.calc_derivatives() # take derivatives as seen above

		# if we need to take a step smaller than 1 (which becomes 1 due to the requirement of parameters being integers), this implies that we have optimized the parameter
		
        if np.abs(self.focus_learning_rate * derivatives["focus"]) > 1:
            self.new_focus = self.focus_history[-1] + self.focus_learning_rate * self.focus_der_history[-1]
            self.new_focus = np.clip(self.new_focus, self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND)
            self.new_focus = round(self.new_focus)
            self.focus_history.append(self.new_focus)
            mirror_values[0] = self.focus_history[-1]
            
        if np.abs(self.second_dispersion_learning_rate * derivatives["second_dispersion"]) > 1:
            self.new_second_dispersion = self.second_dispersion_history[-1] + self.second_dispersion_learning_rate * self.second_dispersion_der_history[-1]
            self.new_second_dispersion = np.clip(self.new_second_dispersion, self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND)
            self.new_second_dispersion = round(self.new_second_dispersion)
            self.second_dispersion_history.append(self.new_second_dispersion)
            dispersion_values[0] = self.second_dispersion_history[-1]

        if np.abs(self.third_dispersion_learning_rate * derivatives["third_dispersion"]) > 1:
            self.new_third_dispersion = self.third_dispersion_history[-1] + self.third_dispersion_learning_rate * self.third_dispersion_der_history[-1]
            self.new_third_dispersion = np.clip(self.new_third_dispersion, self.THIRD_DISPERSION_LOWER_BOUND, self.THIRD_DISPERSION_UPPER_BOUND)
            self.new_third_dispersion = round(self.new_third_dispersion)
            self.third_dispersion_history.append(self.new_third_dispersion)
            dispersion_values[1] = self.third_dispersion_history[-1]
            
        if ( # if there's a repetition in all parameters that means that the change is very small where after rounding it's the same - we have optimized the parameters
            np.abs(self.third_dispersion_learning_rate * derivatives["third_dispersion"]) > 1 and
            np.abs(self.second_dispersion_learning_rate * derivatives["second_dispersion"]) > 1 and
            np.abs(self.focus_learning_rate * derivatives["focus"]) > 1
        ):
            print("convergence achieved")

        if np.abs(self.count_history[-1] - self.count_history[-2]) <= self.tolerance:
            print("convergence achieved")
```

## Communication
After the algorithm has decided on the values, we send the data (`.txt`) to the deformable mirror computer, and Dazzler computer using FTP. 
##### Writing values to Dazzler
- Setup FTP connection (computer connected to network)
- Seed `request.txt` in `D:\GZ0483-HR-670-930p`
- `request.txt` will read from `dispersion.txt` located in the server path `C:\Users\fastlite\Desktop\commands`
##### Sending command
After processing of the data through the algorithm, we send a command txt file to the mirror computer. In order to test the connection between the computers open `cmd` and `ping` the computers.

```python
def upload_files(self):
	mirror_ftp = FTP()
	dazzler_ftp = FTP()
	mirror_ftp.connect(host=self.MIRROR_HOST)
	mirror_ftp.login(user=self.MIRROR_USER, passwd=self.MIRROR_PASSWORD)
	dazzler_ftp.connect(host=self.DAZZLER_HOST)
	dazzler_ftp.login(user=self.DAZZLER_USER, passwd=self.DAZZLER_PASSWORD)
	mirror_files = [os.path.basename(MIRROR_FILE_PATH)]
	dazzler_files = [os.path.basename(DISPERSION_FILE_PATH)]

	for mirror_file_name in mirror_files:
		for dazzler_file_name in dazzler_files:
			focus_file_path = MIRROR_FILE_PATH
			dispersion_file_path = DISPERSION_FILE_PATH
			if os.path.isfile(focus_file_path) and os.path.isfile(dispersion_file_path):
				copy_mirror_IMG_PATH = os.path.join('mirror_command', f'copy_{mirror_file_name}')
				copy_dazzler_IMG_PATH = os.path.join('dazzler_command', f'copy_{dazzler_file_name}')

				try:
					os.makedirs(os.path.dirname(copy_mirror_IMG_PATH))
					os.makedirs(os.path.dirname(copy_dazzler_IMG_PATH))

				except OSError:
					pass

				shutil.copy(focus_file_path, copy_mirror_IMG_PATH)
				shutil.copy(dispersion_file_path, copy_dazzler_IMG_PATH)
				with open(copy_mirror_IMG_PATH, 'rb') as local_file:
					mirror_ftp.storbinary(f'STOR {mirror_file_name}', local_file)
					print(f"Uploaded to mirror FTP: {mirror_file_name}")
				with open(copy_dazzler_IMG_PATH, 'rb') as local_file:
					dazzler_ftp.storbinary(f'STOR {dazzler_file_name}', local_file)
					print(f"Uploaded to dazzler FTP: {dazzler_file_name}")
				os.remove(copy_mirror_IMG_PATH)
				os.remove(copy_dazzler_IMG_PATH)

### Testing description
This is a test for the gradient descent algorithm relying on a definition of an arbitrary function to verify the code correctly finds the maximum. Let's start with testing the code on a convex function, and proceed to test it on more complex functions with local maxima. 

#### Convex parabolic function
For the purpose of testing the `count` was was simulated by the function - no images were processed.

$$C(\phi_{2},\phi_{3},f)=-((\phi_{2}-42)^2 + (\phi_{3}-70)^2 + (f+972)^{2}+3^6)$$
$$\text{Algorithm should arrive at:} \; \phi_{2}=42,\; \phi_{3}=70,\; f=-972$$

in the code this is expressed by the `count_function`:

```python
    def count_function(self, new_focus, new_second_dispersion, new_third_dispersion):
        count_func = -1*(((new_second_dispersion - 42) ** 2) + ((new_third_dispersion - 70) ** 2) + ((new_focus + 972) ** 2)) +3e6
        return count_func
```

where the partial derivatives are calculated:

```python
    def calc_derivatives(self):
        self.count_focus_der = -2*(self.new_focus+972)
        self.count_second_dispersion_der = -2*(self.new_second_dispersion-42)
        self.count_third_dispersion_der = -2*(self.new_third_dispersion-69)
        self.focus_der_history.append(self.count_focus_der)      self.second_dispersion_der_history.append(self.count_second_dispersion_der)
     self.third_dispersion_der_history.append(self.count_third_dispersion_der)
        self.total_gradient = (self.focus_der_history[-1] + self.second_dispersion_der_history[-1] + self.third_dispersion_der_history[-1])
        self.total_gradient_history.append(self.total_gradient)
        self.der_iteration_data.append(self.dir_run_count)
        return {"focus":self.count_focus_der,"second_dispersion":self.count_second_dispersion_der,"third_dispersion":self.count_third_dispersion_der}
```

After running the code the console suggests:
```
initial directions are: focus -1, second_dispersion -1, third_dispersion -1
-------------
convergence achieved
function_value 2053973.0, current values are: focus -197, second_dispersion 6, third_dispersion 13
-------------       
convergence achieved
function_value 2394830.0, current values are: focus -352, second_dispersion 13, third_dispersion 24
-------------       
convergence achieved
function_value 2612643.0, current values are: focus -476, second_dispersion 19, third_dispersion 33
-------------       
convergence achieved
function_value 2752086.0, current values are: focus -575, second_dispersion 24, third_dispersion 40
-------------
convergence achieved
function_value 2982964.0, current values are: focus -868, second_dispersion 37, third_dispersion 62

* * *

-------------
function_value 2999914.0, current values are: focus -967, second_dispersion 37, third_dispersion 64
```

The final line shows we approached `focus -967, second_dispersion 37, third_dispersion 64`. This is due to rounding errors, resulting in the optimized values not being exact, but we get very close. **The optimization algorithm works!**
