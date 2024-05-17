import os
import time
import numpy as np
import pandas as pd
import torch
import torchaudio
from joblib import Parallel, delayed
from torch.utils.data import Dataset

from utils import txt_to_dataframe, bcolors

c = bcolors()


def batch_data(data, max_duration=30):
    batches = []
    current_batch = []

    for entry in data:
        if entry['end_time'] > max_duration * (len(batches) + 1) and entry['start_time'] < max_duration * (
                len(batches) + 2):
            batches.append(current_batch)
            current_batch = []

        current_batch.append(entry)

    if current_batch:
        batches.append(current_batch)

    return batches


# create a single entry for each batch containing all the text separated by a space
def inner_merge_batches(data):
    return [{
        'text': ' '.join([entry['transcript'] for entry in batch]),
        'start': batch[0]['start_time'],
        'end': batch[-1]['end_time']
    } for i, batch in enumerate(data)]


class UKWFunkSprache(Dataset):
    def __init__(self,
                 feed_ids,
                 root_dir,
                 transform=None,
                 n_jobs=-1):
        self.feed_ids = feed_ids
        self.root_dir = root_dir
        self.transform = transform

        print(f"\n{c.OKGREEN}Preloading Samples...{c.ENDC}")
        print(f"\n{c.OKCYAN}Audio Files:         {len(self.feed_ids)}{c.ENDC}")
        print(f"{c.OKCYAN}Jobs:                {n_jobs} {c.ENDC}\n")

        start_time = time.time()
        result = Parallel(n_jobs=n_jobs)(
            delayed(self.process_file)(idx) for idx in range(len(self.feed_ids))
        )

        result = np.concatenate(result)

        self.audio_samples = []
        self.transcriptions = []
        self.speakers = []
        self.groups = []

        for sample_group, sample, speaker, transcript in result:
            self.audio_samples.append(sample)
            self.transcriptions.append(transcript)
            self.speakers.append(speaker)
            self.groups.append(sample_group)

        self.audio_samples = np.array(self.audio_samples)
        end_time = time.time()
        t = end_time - start_time
        print(f"\n{c.OKBLUE}Time taken:      {int((t - (t % 60)) / 60)} min {t % 60} sec {c.ENDC}")

    def process_file(self, idx):
        feed_id = self.feed_ids[idx]
        audio_fpath = self.root_dir + f"audio/{feed_id}.wav"
        text_fpath = self.root_dir + f"text/{feed_id}.csv"

        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_fpath)

        # Load the text transcription file
        transcripts_df = pd.read_csv(text_fpath)

        batches = batch_data(transcripts_df.to_dict('records'))

        metadata = inner_merge_batches(batches)

        sample_group = str(feed_id)
        samples = []
        for i in range(len(metadata)):
            start_time = metadata[i]['start']
            end_time = metadata[i]['end']
            transcript = metadata[i]['text']

            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            sample = waveform[:, start_sample:end_sample].squeeze().numpy()

            samples.append([sample_group, sample, transcript])

        return samples

    def __len__(self):
        return len(self.audio_samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio = self.audio_samples[idx]
        target = self.transcriptions[idx]
        return audio, target
