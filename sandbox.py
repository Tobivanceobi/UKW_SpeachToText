import torch
import zipfile
import torchaudio
from glob import glob

device = torch.device('cpu')

name = 'small_slow'
device = torch.device('cpu')
model, samples, utils = torch.hub.load(
  repo_or_dir='snakers4/silero-models',
  model='silero_denoise',
  name=name,
  device=device)
(read_audio, save_audio, denoise) = utils

audio_path = f'data/17329-20230629-0004.mp3'
audio = read_audio(audio_path).to(device)
output = model(audio)
save_audio(f'data/results.mp3', output.squeeze(1).cpu())

output, sr = denoise(model, f'data/17329-20230629-0004.mp3', f'data/results.mp3', device='cpu')

model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils

test_files = glob('data/17329-20230629-0004.mp3')
print(test_files)
input = prepare_model_input(read_batch(test_files),
                            device=device)
print(input.shape)
output = model(input)
print(output.shape)
for example in output:
    print(example.shape)
    out = decoder(example.cpu())
    print(type(out), out)
