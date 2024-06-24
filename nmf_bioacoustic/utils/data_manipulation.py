def time_to_frame(time_seconds, sr, hl):
    """
    Compute the index of the frame given a time in seconds.

    Parameters:
    - time_seconds: Time in seconds.
    - sr: Sampling rate in Hz.
    - hl: Hop length in samples.

    Returns:
    - frame_index: Index of the frame.
    """
    # Convert time to samples
    samples = time_seconds * sr
    # Compute frame index
    frame_index = int(samples // hl)
    return frame_index

def frame_to_time(frame_index, sr, hl):
    """
    Compute the time in seconds given the frame index.

    Parameters:
    - frame_index: Index of the frame.
    - sr: Sampling rate in Hz.
    - hl: Hop length in samples.

    Returns:
    - time_seconds: Time in seconds.
    """
    # Compute time in samples
    samples = frame_index * hl
    # Convert samples to time
    time_seconds = samples / sr
    return time_seconds

def crop_time(spec, time_limit_s, sr, hl):
    """
    """
    # Compute the number of frames to keep
    limit_frame = time_to_frame(time_limit_s, sr, hl)
    # Crop the spectrogram
    cropped_spec = spec[:, :int(limit_frame)]
    return cropped_spec