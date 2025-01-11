import csv
import os
import json

class SGT:
    def __init__(self, data_handler=None, num_reps=3, rep_time=3, time_between_reps=3, inputs_names=None, output_folder=None):
        
        self.data_handler = data_handler
        self.num_reps = num_reps
        self.rep_time = rep_time
        self.time_between_reps = time_between_reps
        self.output_folder = output_folder
        self.current_input = 0
        self.current_rep = 0
        self.flag = ''
        self.inputs_names = inputs_names.split(",")[:-1]
        self.input_count = len(self.inputs_names)
        self.data = {}
        self.meta_data_dic = {}
    
    def _collect_data(self, flag):
            if (flag == 'S'):
                self.data_handler.raw_data.reset_emg()
                return True
            elif (flag == 'E'):
                self.data[self.current_input] = self.data_handler.get_data()
                self.current_input += 1
                if self.current_input == self.input_count:
                    self.current_input = 0
                return True
            elif (flag == 'R'):
                self._write_data(self.data)
                self.current_rep += 1
                return True
            elif (flag == 'F'):
                return False
    
    def _write_data(self, data):
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder) 
        
        for c in data.keys():
            # Write EMG Files
            emg_file = self.output_folder + "\\R_" + str(self.current_rep) + "_C_" + str(c) + ".csv"
            self.meta_data_dic[emg_file] = {
                'rep_idx': self.num_reps,
                'class_idx': c,
                'class_name': self.inputs_names[c].split(".")[0],
                'file_type': self.inputs_names[c].split(".")[1]
            }
            with open(emg_file, "w", newline='', encoding='utf-8') as file:
                emg_writer = csv.writer(file)
                for row in data[c]:
                    emg_writer.writerow(row)
        # Write Metadata file
        with open(self.output_folder + "\\metadata.json", 'w') as f: 
            f.write(json.dumps(self.meta_data_dic))