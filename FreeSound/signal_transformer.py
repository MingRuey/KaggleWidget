import tensorflow as tf


class TimeSeriesToMel:

    def __init__(
            self,
            target_ms=25,
            raw_freq=44100,
            overlap_ratio=0.5,
            lower_hz=50.0,
            upper_hz=15000.0,
            number_of_mel_bins=256
            ):
        self.target_ms = target_ms
        self.raw_freq = raw_freq
        self.overlap_ratio = overlap_ratio
        self.lower_hz = lower_hz
        self.upper_hz = upper_hz
        self.number_of_mel_bins = number_of_mel_bins

    def get_parser(self):

        frame_length = int(self.target_ms * self.raw_freq / 1000)
        frame_step = int(frame_length * self.overlap_ratio)

        def parser(series):
            series = tf.cast(series, tf.float32)
            stfs = tf.signal.stft(
                series,
                frame_length=frame_length,
                frame_step=frame_step,
                pad_end=True,
                name="stft"
            )

            spectrograms = tf.abs(stfs)

            mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=self.number_of_mel_bins,
                num_spectrogram_bins=stfs.shape[-1],
                sample_rate=self.raw_freq,
                lower_edge_hertz=self.lower_hz,
                upper_edge_hertz=self.upper_hz,
                name="mel_matrix"
            )

            mel_spectrograms = tf.tensordot(
                spectrograms,
                mel_weight_matrix,
                axes=1
            )

            log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

            return log_mel_spectrograms

        return parser


class SpectrogramsToImage:

    def __init__(self, target_height, target_width):
        """Note: H is time diemntsion, W is feature dimentsion"""
        self.h = target_height
        self.w = target_width

    def get_parser(self):

        def parser(spectrograms):
            # expand dimension
            spectrograms = tf.expand_dims(spectrograms, axis=-1)
            image = tf.image.resize_image_with_crop_or_pad(
                spectrograms,
                target_height=self.h,
                target_width=self.w
            )
            return image

        return parser
