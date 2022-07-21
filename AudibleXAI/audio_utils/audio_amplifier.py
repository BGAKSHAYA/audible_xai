from constants import relevant_region_extractor_constants
import wave
import audioop


def amplify_sound(input_file, output_file):
    with wave.open(input_file, 'rb') as wav:
        p = wav.getparams()
        with wave.open(output_file, 'wb') as audio:
            audio.setparams(p)
            frames = wav.readframes(p.nframes)
            audio.writeframesraw(audioop.mul(frames, p.sampwidth,
                                             relevant_region_extractor_constants.REGION_AUDIO_AMPLIFICATION_FACTOR))
