import os
import numpy as np
from ftplib import FTP
import random
from pyqtgraph.Qt import QtCore, QtWidgets
import sys 
import pyqtgraph as pg

MIRROR_FILE_PATH = r'dm_parameters.txt'
DISPERSION_FILE_PATH = r'dazzler_parameters.txt'

with open(MIRROR_FILE_PATH, 'r') as file:
    content = file.read()
mirror_values = list(map(int, content.split()))

with open(DISPERSION_FILE_PATH, 'r') as file:
    content = file.readlines()

dispersion_values = {
    0: int(content[0].split('=')[1].strip()),  # 0 is the key for 'order2'
    1: int(content[1].split('=')[1].strip())   # 1 is the key for 'order3'
}

class BetatronApplication(QtWidgets.QApplication):
    def __init__(self, *args, **kwargs):
        super(BetatronApplication, self).__init__(*args, **kwargs)

        self.new_focus = 0  
        self.new_second_dispersion = 0  
        self.new_third_dispersion = 0  

        self.dir_images_processed = 0
        self.images_processed = 0
        self.count_history = []
        self.focus_learning_rate = 0.1
        self.second_dispersion_learning_rate = 0.1
        self.third_dispersion_learning_rate = 0.1

    # ------------ Plotting ------------ #

        # initialize lists to keep track of optimization process
        self.third_dispersion_der_history = []
        self.second_dispersion_der_history = []
        self.focus_der_history = []
        self.total_gradient_history = []

        self.iteration_data = []
        self.der_iteration_data = []
        
        self.count_plot_widget = pg.PlotWidget()
        self.count_plot_widget.setWindowTitle('Count optimization')
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

        self.initial_focus = mirror_values[0]
        self.focus_history = []    
        # self.FOCUS_LOWER_BOUND = max(self.initial_focus - 20, -200)
        # self.FOCUS_UPPER_BOUND = min(self.initial_focus + 20, 200)

        self.FOCUS_LOWER_BOUND = -999999
        self.FOCUS_UPPER_BOUND = 999999

        self.count_change_tolerance = 10
        
    # ------------ Dazzler ------------ #

        # 36100 initial 
        self.initial_second_dispersion = dispersion_values[0] 
        self.second_dispersion_history = []
        # self.SECOND_DISPERSION_LOWER_BOUND = max(self.initial_second_dispersion - 500, 30000)
        # self.SECOND_DISPERSION_UPPER_BOUND = min(self.initial_second_dispersion + 500, 40000)

        self.SECOND_DISPERSION_LOWER_BOUND = -999999
        self.SECOND_DISPERSION_UPPER_BOUND = 999999

        # -27000 initial
        self.initial_third_dispersion = dispersion_values[1] 
        self.third_dispersion_history = []
        # self.THIRD_DISPERSION_LOWER_BOUND = max(self.initial_third_dispersion -2000, -30000)
        # self.THIRD_DISPERSION_UPPER_BOUND = min(self.initial_third_dispersion + 2000, -25000)

        self.THIRD_DISPERSION_LOWER_BOUND = -999999
        self.THIRD_DISPERSION_UPPER_BOUND = 999999

        self.random_direction = [random.choice([-1, 1]) for _ in range(4)]

    def count_function(self, new_focus, new_second_dispersion, new_third_dispersion):
        count_func = -1*(((new_second_dispersion - 42) ** 2) + ((new_third_dispersion - 70) ** 2) + ((new_focus + 972) ** 2)) +3e6
        return count_func  

    def initial_optimize(self):

        self.new_focus = self.focus_history[-1] + self.random_direction[0]
        self.new_second_dispersion = self.second_dispersion_history[-1] + self.random_direction[1]
        self.new_third_dispersion = self.third_dispersion_history[-1] + self.random_direction[2]
 
        self.new_focus = round(np.clip(self.new_focus, self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND))
        self.new_second_dispersion = round(np.clip(self.new_second_dispersion, self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND))
        self.new_third_dispersion = round(np.clip(self.new_third_dispersion, self.THIRD_DISPERSION_LOWER_BOUND, self.THIRD_DISPERSION_UPPER_BOUND))
 
        self.focus_history.append(self.new_focus)
        self.second_dispersion_history.append(self.new_second_dispersion)
        self.third_dispersion_history.append(self.new_third_dispersion)
 
        mirror_values[0] = (self.focus_history[-1])
        dispersion_values[0] = (self.second_dispersion_history[-1])
        dispersion_values[1] = (self.third_dispersion_history[-1])

    def calc_derivatives(self):
        self.count_focus_der = -2*(self.new_focus+972)
        self.count_second_dispersion_der = -2*(self.new_second_dispersion-42)
        self.count_third_dispersion_der = -2*(self.new_third_dispersion-69)

        self.focus_der_history.append(self.count_focus_der)
        self.second_dispersion_der_history.append(self.count_second_dispersion_der)
        self.third_dispersion_der_history.append(self.count_third_dispersion_der)

        self.total_gradient = (self.focus_der_history[-1] + self.second_dispersion_der_history[-1] + self.third_dispersion_der_history[-1])

        self.total_gradient_history.append(self.total_gradient)
        self.der_iteration_data.append(self.dir_images_processed)
        
        return {
            "focus": self.count_focus_der,
            "second_dispersion": self.count_second_dispersion_der,
            "third_dispersion": self.count_third_dispersion_der
            }
    def optimize_count(self):
        derivatives = self.calc_derivatives()

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
        
        if self.images_processed >2:
            # if the count is not changing much this means that we are near the peak 
            if np.abs(self.count_history[-1] - self.count_history[-2]) <= self.count_change_tolerance:
                print("Convergence achieved")

    def process_images(self):
        self.images_processed += 1
        self.iteration_data.append(self.images_processed)

        if self.images_processed == 1:

            print('-------------')                    
            self.focus_history.append(self.initial_focus)                       
            self.second_dispersion_history.append(self.initial_second_dispersion)
            self.third_dispersion_history.append(self.initial_third_dispersion)      
            print(f"initial directions are: focus {self.random_direction[0]}, second_dispersion {self.random_direction[1]}, third_dispersion {self.random_direction[2]}")
            self.initial_optimize()

        else:
            self.dir_images_processed += 1
            self.img_mean_count = self.count_function(self.new_focus, self.new_second_dispersion, self.new_third_dispersion)
            self.count_history.append(self.img_mean_count)

            self.optimize_count()

            QtCore.QCoreApplication.processEvents()
            print(f"function_value {self.count_history[-1]}, current values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}, third_dispersion {self.third_dispersion_history[-1]}")

        # update the plots

        self.plot_curve.setData(self.der_iteration_data, self.count_history)
        self.total_gradient_curve.setData(self.der_iteration_data, self.total_gradient_history)
        
        # reset variables for next optimization round
        self.image_group_count_sum = 0
        self.mean_count_per_image_group  = 0
        self.img_mean_count = 0  
        print('-------------')

if __name__ == "__main__":
    app = BetatronApplication([])

    for _ in range(30):
        app.process_images()

    win = QtWidgets.QMainWindow()
    sys.exit(app.exec_())