import os
import cv2
import numpy as np
from ftplib import FTP
import shutil
import random
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import sys 
import time 
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

reached_max = {"focus":False,"second_dispersion":False,"third_dispersion":False}

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

        self.count_grad = 0
        self.dir_run_count = 0
        self.run_count = 0
        self.count_history = []
        self.focus_learning_rate = 0.1
        self.second_dispersion_learning_rate = 0.1
        self.third_dispersion_learning_rate = 0.1

    # ------------ Plotting ------------ #

        self.third_dispersion_der_history = []
        self.second_dispersion_der_history = []
        self.focus_der_history = []
        self.total_gradient_history = []

        self.iteration_data = []
        self.der_iteration_data = []
        self.count_data = []
        
        self.count_plot_widget = pg.PlotWidget()
        self.count_plot_widget.setWindowTitle('count optimization')
        self.count_plot_widget.setLabel('left', 'count')
        self.count_plot_widget.setLabel('bottom', 'iteration')
        self.count_plot_widget.showGrid(x=True, y=True)
        self.count_plot_widget.show()

        self.main_plot_window = pg.GraphicsLayoutWidget()
        self.main_plot_window.show()

        layout = self.main_plot_window.addLayout(row=0, col=0)

        self.count_plot_widget = layout.addPlot(title='count vs iteration')
        self.focus_plot = layout.addPlot(title='count_focus_derivative')
        self.second_dispersion_plot = layout.addPlot(title='count_second_dispersion_derivative')
        self.third_dispersion_plot = layout.addPlot(title='count_third_dispersion_derivative')
        self.total_gradient_plot = layout.addPlot(title='total_gradient')

        subplots = [self.count_plot_widget, self.focus_plot, self.second_dispersion_plot, self.third_dispersion_plot, self.total_gradient_plot]
        for subplot in subplots:
            subplot.showGrid(x=True, y=True)

        self.plot_curve = self.count_plot_widget.plot(pen='r')
        self.focus_curve = self.focus_plot.plot(pen='r', name='focus derivative')
        self.second_dispersion_curve = self.second_dispersion_plot.plot(pen='g', name='second dispersion derivative')
        self.third_dispersion_curve = self.third_dispersion_plot.plot(pen='b', name='third dispersion derivative')
        self.total_gradient_curve = self.total_gradient_plot.plot(pen='y', name='total gradient')

        self.plot_curve.setData(self.iteration_data, self.count_history)
        self.focus_curve.setData(self.der_iteration_data, self.focus_der_history)
        self.second_dispersion_curve.setData(self.der_iteration_data, self.second_dispersion_der_history)
        self.third_dispersion_curve.setData(self.der_iteration_data, self.third_dispersion_der_history)
        self.total_gradient_curve.setData(self.der_iteration_data, self.total_gradient_history)

    # ------------ Deformable mirror ------------ #

        # init -150
        self.MIRROR_HOST = "192.168.200.3"
        self.MIRROR_USER = "Utilisateur"
        self.MIRROR_PASSWORD = "alls"    

        self.initial_focus = mirror_values[0]
        self.focus_history = []    
        # self.FOCUS_LOWER_BOUND = max(self.initial_focus - 20, -200)
        # self.FOCUS_UPPER_BOUND = min(self.initial_focus + 20, 200)

        self.FOCUS_LOWER_BOUND = -999999
        self.FOCUS_UPPER_BOUND = 999999

        self.tolerance = 100
        
    # ------------ Dazzler ------------ #

        self.DAZZLER_HOST = "192.168.58.7"
        self.DAZZLER_USER = "fastlite"
        self.DAZZLER_PASSWORD = "fastlite"

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

        self.random_direction = []
        self.random_direction = [random.choice([-1, 1]) for _ in range(4)]

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
        self.der_iteration_data.append(self.dir_run_count)
        
        return {"focus":self.count_focus_der,"second_dispersion":self.count_second_dispersion_der,"third_dispersion":self.count_third_dispersion_der}

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

        if (
            np.abs(self.third_dispersion_learning_rate * derivatives["third_dispersion"]) > 1 and
            np.abs(self.second_dispersion_learning_rate * derivatives["second_dispersion"]) > 1 and
            np.abs(self.focus_learning_rate * derivatives["focus"]) > 1
        ):
            print("convergence achieved")

        # if np.abs(self.count_history[-1] - self.count_history[-2]) <= self.tolerance:
        #     print("meow convergence achieved")

    def process_images(self):
        self.run_count += 1
        self.iteration_data.append(self.run_count)

        if self.run_count == 1:

            print('-------------')                    
            self.focus_history.append(self.initial_focus)                       
            self.second_dispersion_history.append(self.initial_second_dispersion)
            self.third_dispersion_history.append(self.initial_third_dispersion)      
            print(f"initial directions are: focus {self.random_direction[0]}, second_dispersion {self.random_direction[1]}, third_dispersion {self.random_direction[2]}")
            self.initial_optimize()

        else:
            self.dir_run_count += 1
            self.img_mean_count = self.count_function(self.new_focus, self.new_second_dispersion, self.new_third_dispersion)
            self.count_history.append(self.img_mean_count)

            self.optimize_count()

            with open(MIRROR_FILE_PATH, 'w') as file:
                file.write(' '.join(map(str, mirror_values)))

            with open(DISPERSION_FILE_PATH, 'w') as file:
                file.write(f'order2 = {dispersion_values[0]}\n')
                file.write(f'order3 = {dispersion_values[1]}\n')

            QtCore.QCoreApplication.processEvents()
            print(f"function_value {self.count_history[-1]}, current values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}, third_dispersion {self.third_dispersion_history[-1]}")

        # self.upload_files() # send files to second computer

        self.plot_curve.setData(self.der_iteration_data, self.count_history)
        self.focus_curve.setData(self.der_iteration_data, self.focus_der_history)
        self.second_dispersion_curve.setData(self.der_iteration_data, self.second_dispersion_der_history)
        self.third_dispersion_curve.setData(self.der_iteration_data, self.third_dispersion_der_history)
        self.total_gradient_curve.setData(self.der_iteration_data, self.total_gradient_history)

        print('-------------')

if __name__ == "__main__":
    app = BetatronApplication([])

    for _ in range(100):
        app.process_images()

    win = QtWidgets.QMainWindow()
    sys.exit(app.exec_())