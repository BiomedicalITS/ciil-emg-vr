import libemg
import pickle
from libemg.utils import make_regex
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
import time
import copy 

USER_ID = 1
WINDOW_SIZE = 200 #Window size in samples from SGT
WINDOW_INCREMENT = 100
# FEATURE_SET = "LS4" #Feature set selection
NUM_CHANNELS = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
ADAPTATION_EPOCHS = 10
EPOCHS=30


class Memory:
    def __init__(self, max_len=None):
        # What are the current targets for the model?
        self.experience_targets = []
        # What are the inputs for the saved experiences?
        self.experience_data    = []
        # What are the correct options (given the context)
        self.experience_context = []
        # What was the outcome (P or N)
        self.experience_outcome = []
        # What is the group (this is used for CL)
        self.experience_group   = []
        # What is the id of the experience
        self.experience_ids     = []
        # How many memories do we have?
        self.memories_stored = 0
    
    def __len__(self):
        return self.memories_stored
    
    def __add__(self, other_memory):
        assert type(other_memory) == Memory
        if len(other_memory):
            if not len(self):
                self.experience_targets = other_memory.experience_targets
                self.experience_data    = other_memory.experience_data
                self.experience_context = other_memory.experience_context
                self.experience_ids     = other_memory.experience_ids
                self.experience_outcome = other_memory.experience_outcome
                self.experience_group   = other_memory.experience_group
                self.memories_stored    = other_memory.memories_stored
            else:
                self.experience_targets = torch.cat((self.experience_targets,other_memory.experience_targets))
                self.experience_data = torch.vstack((self.experience_data,other_memory.experience_data))
                self.experience_context = np.concatenate((self.experience_context,other_memory.experience_context))
                self.experience_ids.extend(list(range(self.memories_stored, self.memories_stored + other_memory.memories_stored)))
                self.experience_outcome = np.concatenate((self.experience_outcome, other_memory.experience_outcome)) 
                self.experience_group   = np.concatenate((self.experience_group, other_memory.experience_group))
                self.memories_stored += other_memory.memories_stored
        return self
        
    
    def add_memories(self, experience_data, experience_targets, experience_context, experience_outcome, experience_group):
        if len(experience_targets):
            if not len(self):
                self.experience_targets = experience_targets
                self.experience_data    = experience_data
                self.experience_context = experience_context
                self.experience_ids     = list(range(len(experience_targets)))
                self.experience_outcome = experience_outcome
                self.experience_group   = experience_group
                self.memories_stored    += len(experience_targets)
            else:
                self.experience_targets = torch.cat((self.experience_targets,experience_targets))
                self.experience_data = torch.vstack((self.experience_data,experience_data))
                self.experience_context = np.concatenate((self.experience_context,experience_context))
                self.experience_ids.extend(list(range(self.memories_stored, self.memories_stored + len(experience_targets))))
                self.experience_outcome = np.concatenate((self.experience_outcome, experience_outcome)) 
                self.experience_group   = np.concatenate((self.experience_group, experience_group))
                self.memories_stored += len(experience_targets)
    
    def shuffle(self):
        if len(self):
            indices = list(range(len(self)))
            random.shuffle(indices)
            # shuffle the keys
            self.experience_targets = self.experience_targets[indices]
            self.experience_data    = self.experience_data[indices]
            # SGT does not have these fields
            if len(self.experience_context):
                self.experience_context = self.experience_context[indices]
                self.experience_ids     = [self.experience_ids[i] for i in indices]
                self.experience_outcome = [self.experience_outcome[i] for i in indices]
                self.experience_group   = [self.experience_group[i] for i in indices]

        
    def unshuffle(self):
        unshuffle_ids = [i[0] for i in sorted(enumerate(self.experience_ids), key=lambda x:x[1])]
        if len(self):
            self.experience_targets = self.experience_targets[unshuffle_ids]
            self.experience_data    = self.experience_data[unshuffle_ids]
            # SGT does not have these fields
            if len(self.experience_context):
                self.experience_context = self.experience_context[unshuffle_ids]
                self.experience_outcome = [self.experience_outcome[i] for i in unshuffle_ids]
                self.experience_group   = [self.experience_group[i] for i in unshuffle_ids]
                self.experience_ids     = [self.experience_ids[i] for i in unshuffle_ids]

    def write(self, save_dir, num_written=""):
        with open(save_dir + f'classifier_memory_{num_written}.pkl', 'wb') as handle:
            pickle.dump(self, handle)
    
    def read(self, save_dir):
        with open(save_dir +  'classifier_memory.pkl', 'rb') as handle:
            loaded_content = pickle.load(self, handle)
            self.experience_targets = loaded_content.experience_targets
            self.experience_data    = loaded_content.experience_data
            self.experience_context = loaded_content.experience_context
            self.experience_outcome = loaded_content.experience_outcome
            self.experience_group   = loaded_content.experience_group
            self.experience_ids     = loaded_content.experience_ids
            self.memories_stored = loaded_content.memories_stored
    
    def from_file(self, save_dir, memory_id):
        with open(save_dir + f'classifier_memory_{memory_id}.pkl', 'rb') as handle:
            obj = pickle.load(handle)
        self.experience_targets = obj.experience_targets
        self.experience_data    = obj.experience_data
        self.experience_context = obj.experience_context
        self.experience_outcome = obj.experience_outcome
        self.experience_group   = obj.experience_group
        self.experience_ids     = obj.experience_ids
        self.memories_stored    = obj.memories_stored


class Config:
    def __init__(self):
        # Change this for each participant:
        self.subjectID = 0
        # Change this for each trial:
        self.trial = 0



def make_base_model(odh):
    config = Config()
    savedir = f"Data/subject{config.subjectID}/trial{config.trial}/"
    savedir = f"Data/"
    fe = libemg.feature_extractor.FeatureExtractor()
    
    feature_list = ["MAV"]#fe.get_feature_groups()[FEATURE_SET]
    offline_classifier = libemg.emg_classifier.EMGClassifier()
    mdl = MLP(input_shape = len(feature_list)*NUM_CHANNELS)

    offline_classifier.__setattr__("classifier", mdl)
    # this is a initialization for speed. 
    # TODO: fix this
    offline_classifier.__setattr__("velocity", True)
    th_min_dic = {}
    th_max_dic = {}
    for i in range(9):
        th_min_dic[i] = 0 # random guesses -- velocity isn't important
        th_max_dic[i] = 100
    offline_classifier.__setattr__("th_min_dic", th_min_dic)
    offline_classifier.__setattr__("th_max_dic", th_max_dic)
    '''
    classifier = libemg.emg_classifier.OnlineEMGClassifier(offline_classifier=offline_classifier,
                                                                window_size=WINDOW_SIZE,
                                                                window_increment=WINDOW_INCREMENT,
                                                                online_data_handler=odh,
                                                                features=feature_list,
                                                                save_dir = savedir,
                                                                save_predictions=True,
                                                                output_format="probabilities",
                                                                std_out=True)
    '''
    classifier = libemg.emg_classifier.OnlineEMGClassifier(offline_classifier=offline_classifier,
                                                                window_size=WINDOW_SIZE,
                                                                window_increment=WINDOW_INCREMENT,
                                                                online_data_handler=odh,
                                                                features=feature_list,
                                                                output_format="probabilities",
                                                                std_out=True)    
    classifier.classifier.classifier.models["live"]["classifier"].eval()
    classifier.run(block=False)
    return classifier


def use_sgt_for_model(num_reps, num_inputs, output_folder=None, oc=None):
    config = Config()
    SGT_save_dir = f"Data/subject{config.subjectID}/trial{config.trial}"
    # Step 1: Parse offline training data
    classes_values = [str(i) for i in range(num_inputs)]
    classes_regex = make_regex(left_bound = "C_", right_bound=".csv", values = classes_values)
    reps_values = [str(i) for i in range(num_reps)]
    reps_regex = make_regex(left_bound = "R_", right_bound="_C", values = reps_values)
    dic = {
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex
    }

    odh = libemg.data_handler.OfflineDataHandler()
    odh.get_data(folder_location=output_folder, filename_dic=dic, delimiter=",")
    train_windows, train_metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    fe = libemg.feature_extractor.FeatureExtractor()
    feature_list = ["MAV"]#fe.get_feature_groups()[config.features]
    features = fe.extract_features(feature_list, train_windows)

    # fe.visualize_feature_space(features, "PCA",train_metadata["classes"])

    features = torch.hstack([torch.tensor(features[key], dtype=torch.float32) for key in features.keys()])
    targets = torch.tensor(train_metadata["classes"], dtype=torch.long)
    offline_classifier = libemg.emg_classifier.EMGClassifier()
    mdl = MLP(input_shape = len(feature_list)*NUM_CHANNELS)
    mdl.memory.experience_data = features
    mdl.memory.experience_targets = targets
    mdl.memory.experience_targets = torch.vstack([torch.eye(9, dtype=torch.float32)[i,:] for i in mdl.memory.experience_targets.tolist()])
    mdl.memory.memories_stored = features.shape[0]
    mdl.train(EPOCHS, shuffle_every_epoch=True)
    trained_weights = mdl.models["background"]["classifier"].state_dict()

    oc.classifier.classifier.models["background"]["classifier"].load_state_dict(trained_weights)
    oc.classifier.classifier.models["live"]["classifier"].load_state_dict(trained_weights)
    oc.raw_data.set_classifier(oc.classifier.classifier)
    print("completed SGT training")
    return True

class MLP(nn.Module):
    def __init__(self, input_shape):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.models = {}

        self.setup_model("live")
        self.setup_model("background")

        self.overwrite_model("live", "background")
        self.send_model_to("live","cpu")
        self.send_model_to("background","cpu")
        
        
        self.setup_optimizers("background")
        
        self.memory = Memory()
        self.batch_size = BATCH_SIZE
        self.frames_saved = 0

    def setup_model(self, name="live"):
        self.models[name] = {}
        self.models[name]["device"] = "cpu" 
        # The encoder is the same shape as the MLP's encoder:
        self.models[name]["classifier"] = nn.Sequential(
            nn.Linear(self.input_shape, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 9),
            nn.Softmax(dim=1)
        )
        
    def send_model_to(self, name, where):
        for key in self.models[name].keys():
            if hasattr(self.models[name][key],"eval"):
                self.models[name][key].to(where)

    def setup_optimizers(self, name="background"):
        # set optimizer
        self.optimizer_classifier = optim.Adam(self.models[name]["classifier"].parameters(), lr=LEARNING_RATE)
        self.loss_function = nn.MSELoss()
    
    def overwrite_model(self, to_be_overwritten="live", overwrite_with="background"):
        self.models[to_be_overwritten] = copy.deepcopy(self.models[overwrite_with])

    def update_live_model(self):
        self.overwrite_model()
        self.send_model_to("live","cpu")
        self.models["live"]["classifier"].eval()
        

    def normalize(self, x):
        # if len(x):
        #     return (x - means)/stds
        return x
    
    def forward(self,x,name="live"):
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        x.requires_grad=False
        x = self.normalize(x)
        return self.models[name]["classifier"](x)
    
    def forward_reconstruct(self,x,name="background"):
        # to get a peek at the higher dimensional space
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        x.requires_grad=False
        x = self.normalize(x)
        x = self.models[name]["classifier"][0](x)
        x = self.models[name]["classifier"][1](x)
        x = self.models[name]["classifier"][2](x)
        x = self.models[name]["classifier"][3](x)
        x = self.models[name]["classifier"][4](x)
        return x
        
    def predict(self, data):
        probs = self.predict_proba(data)
        return np.array([np.where(p==np.max(p))[0][0] for p in probs])

    def predict_proba(self,data):
        if type(data) == np.ndarray:
            data = torch.tensor(data, dtype=torch.float32)
        output = self.forward(data, "live")
        return output.detach().numpy()

    def adapt(self, memory):
        self.memory = memory
        self.train()

        if len(self.memory) > 2:
            self.baseline_save()

    def train(self, epochs=ADAPTATION_EPOCHS, shuffle_every_epoch=True):
        num_batch = len(self.memory) // self.batch_size
        t = time.time()
        losses = []
        for e in range(epochs):
            if shuffle_every_epoch:
                self.memory.shuffle()
            loss = []
            batch_start = 0
            if num_batch > 0:
                for b in range(num_batch):
                    batch_end = batch_start + self.batch_size
                    self.optimizer_classifier.zero_grad()
                    predictions = self.forward(self.memory.experience_data[batch_start:batch_end,:], name="background")
                    loss_value = self.loss_function(predictions, self.memory.experience_targets[batch_start:batch_end])
                    loss_value.backward()
                    self.optimizer_classifier.step()
                    loss.append(loss_value.item())
                    batch_start = batch_end
                losses.append(sum(loss)/len(loss))
                print(f"E {e}: loss: {losses[-1]:.2f}")
            print("-"*15)
        elapsed_time = time.time() - t
        print(f"Adaptation_time = {elapsed_time:.2f}s" )



if __name__ == "__main__":
    config = Config()
    p = libemg.streamers.emager_streamer() #process to start myo giving out data
    odh = libemg.data_handler.OnlineDataHandler() #online data handler: process to start grabbing myo data
    odh.start_listening()
    oc = make_base_model(odh)
    time.sleep(3)
    oc.pause()
    SGT_save_dir = f"Data/subject{config.subjectID}/trial{config.trial}"
    use_sgt_for_model(3, 9, SGT_save_dir, oc)
    oc.resume()
    count = 0
    while True:
        time.sleep(1)
        print(count)
        count += 1
        