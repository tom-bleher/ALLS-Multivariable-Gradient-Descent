import os
import cv2
import numpy as np
from ftplib import FTP
import shutil
import random
from watchdog.events import FileSystemEventHandler
from pyqtgraph.Qt import QtCore, QtWidgets
import sys 
import pyqtgraph as pg
from watchdog.observers import Observer

# the txt files the code adjusts and uploads 
MIRROR_FILE_PATH = r'dm_parameters.txt'
DISPERSION_FILE_PATH = r'dazzler_parameters.txt'

# open and read the txt files and read the initial values
with open(MIRROR_FILE_PATH, 'r') as file:
    content = file.read()
mirror_values = list(map(int, content.split()))

with open(DISPERSION_FILE_PATH, 'r') as file:
    content = file.readlines()

dispersion_values = {
    0: int(content[0].split('=')[1].strip()),  # 0 is the key for 'order2'
    1: int(content[1].split('=')[1].strip())   # 1 is the key for 'order3'
}

class ImageHandler(FileSystemEventHandler):
    def __init__(self, process_images_callback):
        super().__init__()
        self.process_images_callback = process_images_callback

    def on_created(self, event):
        if not event.is_directory:
            self.process_images_callback([event.src_path])
                      
class BetatronApplication(QtWidgets.QApplication):
    def __init__(self, *args, **kwargs):
        super(BetatronApplication, self).__init__(*args, **kwargs)

        # for how many images should the mean be taken for
        self.image_group = 2

        self.mean_count_per_image_group  = 0
        self.image_groups_dir_run_count = 0

        # keep track of number of processed images and image groups
        self.image_groups_processed = 0
        self.images_processed = 0

        self.image_group_count_sum = 0  
        self.count_history = np.array([])

        # set learning rates for the different optimization variables
        self.focus_learning_rate = 0.1
        self.second_dispersion_learning_rate = 0.1
        self.third_dispersion_learning_rate = 0.1

        # image path (should match to path specified in SpinView)
        self.IMG_PATH = r'images'

        # setup tracking for new images
        self.waiting_for_images_printed = False
        self.initialize_image_files()

    # ------------ Plotting ------------ #

        # initialize lists to keep track of optimization process
        self.third_dispersion_der_history = np.array([])
        self.second_dispersion_der_history = np.array([])
        self.focus_der_history = np.array([])
        self.total_gradient_history = np.array([])

        self.iteration_data = np.array([])
        self.der_iteration_data = np.array([])
        
        self.count_plot_widget = pg.PlotWidget()
        self.count_plot_widget.setWindowTitle('count optimization')
        self.count_plot_widget.setLabel('left', 'Count')
        self.count_plot_widget.setLabel('bottom', 'Image group iteration')
        self.count_plot_widget.show()

        self.main_plot_window = pg.GraphicsLayoutWidget()
        self.main_plot_window.show()

        layout = self.main_plot_window.addLayout(row=0, col=0)

        self.count_plot_widget = layout.addPlot(title='Count vs image group iteration')
        self.total_gradient_plot = layout.addPlot(title='Total gradient vs image group iteration')

        self.plot_curve = self.count_plot_widget.plot(pen='r')
        self.total_gradient_curve = self.total_gradient_plot.plot(pen='y', name='total gradient')\
        
        # y labels of plots
        self.total_gradient_plot.setLabel('left', 'Total Gradient')
        self.count_plot_widget.setLabel('left', 'Image Group Iteration')

        # x label of both plots
        self.count_plot_widget.setLabel('bottom', 'Image Group Iteration')
        self.total_gradient_plot.setLabel('bottom', 'Image Group Iteration')

        self.plot_curve.setData(self.iteration_data, self.count_history)
        self.total_gradient_curve.setData(self.der_iteration_data, self.total_gradient_history)

    # ------------ Deformable mirror ------------ #

        # init -150
        # connect to the mirror
        #self.mirror_ftp = FTP()
        #self.mirror_ftp.connect(MIRROR_HOST="192.168.200.3")
        #self.mirror_ftp.login(MIRROR_USER="Utilisateur", MIRROR_PASSWORD="alls")

        # set initial focus value from txt flle and initialize focus history list
        self.initial_focus = mirror_values[0]
        self.focus_history = np.array([], dtype=int)    
        
        # define global and local bounds for the deformable mirror 
        self.FOCUS_LOWER_BOUND = max(self.initial_focus - 20, -200)
        self.FOCUS_UPPER_BOUND = min(self.initial_focus + 20, 200)

        # set count change tolerance under which the program will consider the case optimized 
        self.count_change_tolerance = 10
        
    # ------------ Dazzler ------------ #

        # setup ftp connection to dazzler
        #self.dazzler_ftp = FTP()
        #self.dazzler_ftp.connect(MIRROR_HOST="192.168.58.7")
        #self.dazzler_ftp.login(MIRROR_USER="fastlite", MIRROR_PASSWORD="fastlite")

        # 36100 initial 
        self.initial_second_dispersion = dispersion_values[0] 
        self.second_dispersion_history = np.array([], dtype=int)

        # define global and local bounds for the dazzler
        self.SECOND_DISPERSION_LOWER_BOUND = max(self.initial_second_dispersion - 500, 30000)
        self.SECOND_DISPERSION_UPPER_BOUND = min(self.initial_second_dispersion + 500, 40000)

        # -27000 initial
        self.initial_third_dispersion = dispersion_values[1] 
        self.third_dispersion_history = np.array([], dtype=int)
        self.THIRD_DISPERSION_LOWER_BOUND = max(self.initial_third_dispersion -2000, -30000)
        self.THIRD_DISPERSION_UPPER_BOUND = min(self.initial_third_dispersion + 2000, -25000)

        self.random_direction = np.array([])

        self.image_handler = ImageHandler(self.process_images)
        self.file_observer = Observer()
        self.file_observer.schedule(self.image_handler, path=self.IMG_PATH, recursive=False)
        self.file_observer.start()

        self.random_direction = [random.choice([-1, 1]) for _ in range(4)]

            
    def initialize_image_files(self):
        if not self.waiting_for_images_printed:
            print("Waiting for images ...")
            self.waiting_for_images_printed = True
        
        # define a list to store the paths of new files
        self.new_files = [] 
        
        # iterate over each filename in the IMG_PATH directory
        for filename in os.listdir(self.IMG_PATH):
            # check if the filename ends with '.tiff'
            if filename.endswith('.tiff'):
                # add the file's path to the new_files list
                self.new_files.append(os.path.join(self.IMG_PATH, filename))

        if self.new_files:
            self.image_files = self.new_files
            
    # method used to send the new values to the mirror and dazzler computers via FTP
    def upload_files(self):
 
        mirror_files = [os.path.basename(MIRROR_FILE_PATH)]
        dazzler_files = [os.path.basename(DISPERSION_FILE_PATH)]

        # try to send the file via ftp connection
        try:

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
                            self.mirror_ftp.storbinary(f'STOR {mirror_file_name}', local_file)
                            print(f"Uploaded to mirror FTP: {mirror_file_name}")

                        with open(copy_dazzler_IMG_PATH, 'rb') as local_file:
                            self.dazzler_ftp.storbinary(f'STOR {dazzler_file_name}', local_file)
                            print(f"Uploaded to dazzler FTP: {dazzler_file_name}")

                        os.remove(copy_mirror_IMG_PATH)
                        os.remove(copy_dazzler_IMG_PATH)

        except Exception as e:
            print(f"Error in FTP upload: {e}")
    
    # method to calculate count (by its brightness proxy)   
    def calc_count_per_image(self, image_path):
    
        # read the image in 16 bit
        original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        
        # apply median blur on image
        median_blured_image = cv2.medianBlur(original_image, 5)
        
        # calculate mean brightness of blured image
        self.single_self.img_mean_count = median_blured_image.mean()
        
        # return the count (brightness of image)
        return self.single_self.img_mean_count

    # initial method to start optimization process
    def initial_optimize(self):

        # take random direction for each of the variables
        self.new_focus = self.focus_history[-1] + self.random_direction[0]
        self.new_second_dispersion = self.second_dispersion_history[-1] + self.random_direction[1]
        self.new_third_dispersion = self.third_dispersion_history[-1] + self.random_direction[2]
 
        # the values have to be rounded and clipped due to physical constraints
        self.new_focus = int(round(np.clip(self.new_focus, self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND)))
        self.new_second_dispersion = int(round(np.clip(self.new_second_dispersion, self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND)))
        self.new_third_dispersion = int(round(np.clip(self.new_third_dispersion, self.THIRD_DISPERSION_LOWER_BOUND, self.THIRD_DISPERSION_UPPER_BOUND)))

        # add variables to respective lists
        self.focus_history = np.append(self.focus_history, [self.new_focus])
        self.second_dispersion_history = np.append(self.second_dispersion_history, [self.new_second_dispersion])
        self.third_dispersion_history = np.append(self.third_dispersion_history, [self.new_third_dispersion])

        mirror_values[0] = self.new_focus
        dispersion_values[0] = self.new_second_dispersion
        dispersion_values[1] = self.new_third_dispersion

    def calc_derivatives(self):
        
        # take derivative for every parameter
        self.count_focus_der = (self.count_history[-1] - self.count_history[-2]) / (self.focus_history[-1] - self.focus_history[-2])
        self.count_second_dispersion_der = (self.count_history[-1] - self.count_history[-2]) / (self.second_dispersion_history[-1] - self.second_dispersion_history[-2])
        self.count_third_dispersion_der = (self.count_history[-1] - self.count_history[-2]) / (self.third_dispersion_history[-1] - self.third_dispersion_history[-2])

        # add the derivatives to according history lists for plotting
        self.focus_der_history = np.append(self.focus_der_history, [self.count_focus_der])
        self.second_dispersion_der_history = np.append(self.second_dispersion_der_history, [self.count_second_dispersion_der])
        self.third_dispersion_der_history = np.append(self.third_dispersion_der_history, [self.count_third_dispersion_der])

        # add all derivatives for different parameters
        self.total_gradient = (self.focus_der_history[-1] + self.second_dispersion_der_history[-1] + self.third_dispersion_der_history[-1])

        # add to respective lists for plotting and tracking
        self.total_gradient_history = np.append(self.total_gradient_history, self.total_gradient)
        self.der_iteration_data = np.append(self.der_iteration_data, self.image_groups_dir_run_count)

        # return dicts with derivatives
        return {
            "focus":self.count_focus_der,
            "second_dispersion":self.count_second_dispersion_der,
            "third_dispersion":self.count_third_dispersion_der
            }

    # main optimization block for gradient descent 
    def optimize_count(self):
        
        # get count derivatives for parameters
        derivatives = self.calc_derivatives()
        
        if np.abs(self.focus_learning_rate * derivatives["focus"]) > 1:
            self.new_focus = self.focus_history[-1] + self.focus_learning_rate * self.focus_der_history[-1]
            self.new_focus = np.clip(self.new_focus, self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND)
            self.new_focus = int(round(self.new_focus))

            self.focus_history = np.append(self.focus_history, self.new_focus)
            mirror_values[0] = self.new_focus

        if np.abs(self.second_dispersion_learning_rate * derivatives["second_dispersion"]) > 1:
            self.new_second_dispersion = self.second_dispersion_history[-1] + self.second_dispersion_learning_rate * self.second_dispersion_der_history[-1]
            self.new_second_dispersion = np.clip(self.new_second_dispersion, self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND)
            self.new_second_dispersion = int(round(self.new_second_dispersion))

            self.second_dispersion_history = np.append(self.second_dispersion_history, self.new_second_dispersion)
            dispersion_values[0] = self.new_second_dispersion

        if np.abs(self.third_dispersion_learning_rate * derivatives["third_dispersion"]) > 1:
            self.new_third_dispersion = self.third_dispersion_history[-1] + self.third_dispersion_learning_rate * self.third_dispersion_der_history[-1]
            self.new_third_dispersion = np.clip(self.new_third_dispersion, self.THIRD_DISPERSION_LOWER_BOUND, self.THIRD_DISPERSION_UPPER_BOUND)
            self.new_third_dispersion = int(round(self.new_third_dispersion))

            self.third_dispersion_history = np.append(self.third_dispersion_history, self.new_third_dispersion)
            dispersion_values[1] = self.new_third_dispersion
        
        # if the change in all variables is less than one (we can not take smaller steps thus this is the optimization boundry)
        if (
            np.abs(self.third_dispersion_learning_rate * derivatives["third_dispersion"]) < 1 and
            np.abs(self.second_dispersion_learning_rate * derivatives["second_dispersion"]) < 1 and
            np.abs(self.focus_learning_rate * derivatives["focus"]) < 1
        ):
            print("Convergence achieved")
            
        # stop optimizing parameter if we reached optimization resolution limit
        
        elif np.abs(self.third_dispersion_learning_rate * derivatives["third_dispersion"]) < 1:
            print("Convergence achieved in third dispersion")
        
        elif np.abs(self.second_dispersion_learning_rate * derivatives["second_dispersion"]) < 1:
            print("Convergence achieved in second dispersion")
            
        elif np.abs(self.focus_learning_rate * derivatives["focus"]) < 1:
            print("Convergence achieved in focus")
        
        # if the count is not changing much this means that we are near the peak 
        if np.abs(self.count_history[-1] - self.count_history[-2]) <= self.count_change_tolerance:
            print("Convergence achieved")

    def process_images(self, new_images):
        self.initialize_image_files()
        
        new_images = [image_path for image_path in new_images if os.path.exists(image_path)]
        new_images.sort(key=os.path.getctime)
        
        # loop over all new images 
        for image_path in new_images:
            self.img_mean_count = self.calc_count_per_image(image_path)
            self.image_group_count_sum += np.sum(self.img_mean_count)

            # keep track of the times the program ran (number of images we processed)
            self.images_processed += 1

            # conditional to check if the desired numbers of images to mean was processed
            if self.images_processed % self.image_group == 0:
                # take the mean count for the number of images set
                self.mean_count_per_image_group = np.mean(self.img_mean_count)
                # append to count_history list to keep track of count through the optimization process
                self.count_history = np.append(self.count_history, self.mean_count_per_image_group)

                # update count for 'images_group' processed (number of image groups processed)
                self.image_groups_processed += 1
                self.iteration_data = np.append(self.iteration_data, self.image_groups_processed)

                # if we are in the first time where the algorithm needs to adjust the value
                if self.image_groups_processed == 1:
                    print('-------------')       

                    # add initial values to lists
                    self.focus_history = np.append(self.focus_history, self.initial_focus)      
                    self.second_dispersion_history = np.append(self.second_dispersion_history, self.initial_second_dispersion)                   
                    self.third_dispersion_history = np.append(self.third_dispersion_history, self.initial_third_dispersion)
                    
                    # print to help track the evolution of the system
                    print(f"initial values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}, third_dispersion {self.third_dispersion_history[-1]}")
                    print(f"initial directions are: focus {self.random_direction[0]}, second_dispersion {self.random_direction[1]}, third_dispersion {self.random_direction[2]}")
                    
                    # call function to take random directions
                    self.initial_optimize()

                else:
                    self.image_groups_dir_run_count += 1
                    self.optimize_count()

                # write values to text files
                with open(MIRROR_FILE_PATH, 'w') as file:
                    file.write(' '.join(map(str, mirror_values)))

                with open(DISPERSION_FILE_PATH, 'w') as file:
                    file.write(f'order2 = {dispersion_values[0]}\n')
                    file.write(f'order3 = {dispersion_values[1]}\n')

                QtCore.QCoreApplication.processEvents()

                # print the latest mean count (helps track system)
                print(f"Mean count for last {self.image_group} images: {self.count_history[-1]:.2f}")

                # print the current parameter values which resulted in the brightness above
                print(f"Current values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}, third_dispersion {self.third_dispersion_history[-1]}")
                
                # after the algorithm adjusted the value and wrote it to the txt, send new txt to deformable mirror computer
                # self.upload_files()
                
                # update the plots
                self.plot_curve.setData(self.iteration_data, self.count_history)
                self.total_gradient_curve.setData(self.der_iteration_data, self.total_gradient_history)

                # reset variables for next optimization round
                self.image_group_count_sum = 0
                self.mean_count_per_image_group  = 0
                self.img_mean_count = 0  
                print('-------------')

if __name__ == "__main__":
    app = BetatronApplication([])
    win = QtWidgets.QMainWindow()
    sys.exit(app.exec_())