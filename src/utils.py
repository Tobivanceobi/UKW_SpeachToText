import pandas as pd
import pickle


def load_object(fname):
    try:
        with open(fname + ".pickle", "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def save_object(obj, fname):
    try:
        with open(fname + ".pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def txt_to_dataframe(file_path):
    # Initialize lists to hold the parsed data
    start_times = []
    end_times = []
    speakers = []
    transcripts = []

    # Open the text file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components
            start_time, end_time, temp = line.strip().split('\t')
            speaker, transcript = temp.split(':')

            # Append parsed data to lists
            start_times.append(float(start_time))
            end_times.append(float(end_time))
            speakers.append(speaker)
            transcripts.append(transcript)

    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'start_time': start_times,
        'end_time': end_times,
        'speaker': speakers,
        'transcript': transcripts
    })
    return df
