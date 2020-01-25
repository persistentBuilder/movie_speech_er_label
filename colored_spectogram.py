import wave
import pylab


def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    #pylab.figure(num=None, figsize=(19, 12))
    #pylab.subplot(111)
    #pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    spec_img_path = wav_file.split('.')[0]+'.png'
    pylab.savefig(spec_img_path)
    return spec_img_path

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    print(sound_info.shape)
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate
