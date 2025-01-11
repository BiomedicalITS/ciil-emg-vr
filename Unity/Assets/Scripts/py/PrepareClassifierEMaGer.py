import libemg
from libemg.utils import make_regex


USER_ID = 2
WINDOW_SIZE = 40 #Window size in samples from SGT
WINDOW_INCREMENT = 20
FEATURE_SET = "LS4" #Feature set selection

'''
def collect_data():
    p = libemg.streamers.emager_streamer() #process to start myo giving out data
    odh = libemg.data_handler.OnlineDataHandler() #online data handler: process to start grabbing myo data
    odh.start_listening()
    # Launch training ui
    training_ui = libemg.screen_guided_training.ScreenGuidedTraining()
    training_ui.download_gestures([1,2,3,4,5,], "images/") #downloads images from github repo
    training_ui.launch_training(odh, 5, 3, "images/", "Data/subject" + str(USER_ID) +"/SGT/", 3)# 
    p.kill()
'''
def prepare_classifier(num_reps, num_inputs, output_folder):
    #Step 1: Parse offline training data
    #dataset_folder = 'Data/subject' + str(USER_ID) + '/SGT/'
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

    offline_dh = libemg.data_handler.OfflineDataHandler()
    offline_dh.get_data(folder_location=output_folder, filename_dic=dic, delimiter=",")
    train_windows, train_metadata = offline_dh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)

    # Step 2: Extract features from offline data
    fe = libemg.feature_extractor.FeatureExtractor()
    feature_list = fe.get_feature_groups()[FEATURE_SET]
    training_features = fe.extract_features(feature_list, train_windows)

    # Step 3: Dataset creation
    data_set = {}
    data_set['training_features'] = training_features
    data_set['training_labels'] = train_metadata['classes']

    # Step 4: Create online EMG classifier and start classifying.
    offline = libemg.emg_classifier.EMGClassifier()
    offline.fit("LDA", feature_dictionary=data_set)
    offline.add_majority_vote(5)
    
    return offline

def start_live_classifier(offline_classifier, onlinedh):
    fe = libemg.feature_extractor.FeatureExtractor()

    feature_list = fe.get_feature_groups()[FEATURE_SET]
    classifier = libemg.emg_classifier.OnlineEMGClassifier(offline_classifier, window_size=WINDOW_SIZE, window_increment=WINDOW_INCREMENT, 
                    online_data_handler=onlinedh, features=feature_list, std_out=True, port=12347)
    classifier.run(block=False)
    return classifier

