"""Library for image and video processing using ffmpeg."""

import os
import numpy as np
import ffmpeg


def save_to_movie(out_path, frame_path_format, fps=30, start_frame=0):
    """Creates an mp4 video clip by using already stored frames.

    Args:
      out_path: The video path to store.
      frame_path_format: The image frames format. e.g., <frames_path>/%05d.jpg
      fps: FPS for the output video.
      start_frame: The start frame index.
    """
    # create movie and save it to destination
    command = [
        "ffmpeg",
        "-start_number",
        str(start_frame),
        "-framerate",
        str(fps),
        "-r",
        str(fps),  # output is 30 fps
        "-loglevel",
        "panic",
        "-i",
        frame_path_format,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-profile:v",
        "high",
        "-level:v",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-y",
        out_path,
    ]
    os.system(" ".join(command))


def attach_audio_to_movie(video_path, audio_path, out_path):
    """Attach audio(wav) to video(mp4)."""
    command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-strict",
        "-2",
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-strict",
        "-2",
        "-c:a",
        "aac",
        out_path,
        "-shortest",
    ]
    os.system(" ".join(command))


def hstack_movies(video_path1, video_path2, out_path):
    """Stack two same-height videos horizontally."""
    command = [
        "ffmpeg",
        "-i",
        str(video_path1),
        "-i",
        str(video_path2),
        "-filter_complex",
        "hstack",
        out_path,
    ]
    os.system(" ".join(command))


def stack_movies3x3(video_paths, out_path):
    """Stack 9 same-size videos into a 3x3 grid video."""
    command = ["ffmpeg"]
    for video_path in video_paths:
        command += ["-i", str(video_path)]
    command += [
        "-filter_complex",
        '"[0:v] setpts=PTS-STARTPTS, scale=qvga [a0]; '
        + "[1:v] setpts=PTS-STARTPTS, scale=qvga [a1]; "
        + "[2:v] setpts=PTS-STARTPTS, scale=qvga [a2]; "
        + "[3:v] setpts=PTS-STARTPTS, scale=qvga [a3]; "
        + "[4:v] setpts=PTS-STARTPTS, scale=qvga [a4]; "
        + "[5:v] setpts=PTS-STARTPTS, scale=qvga [a5]; "
        + "[6:v] setpts=PTS-STARTPTS, scale=qvga [a6]; "
        + "[7:v] setpts=PTS-STARTPTS, scale=qvga [a7]; "
        + "[8:v] setpts=PTS-STARTPTS, scale=qvga [a8]; "
        + "[a0][a1][a2][a3][a4][a5][a6][a7][a8]xstack=inputs=9:"
        + 'layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0|0_h0+h1|w0_h0+h1|w0+w1_h0+h1[out]"',
        "-map",
        '"[out]"',
        str(out_path),
    ]
    os.system(" ".join(command))


def ffmpeg_video_read(video_path, fps=None):
    """Video reader based on FFMPEG.

    This function supports setting fps for video reading.
    The FPS can be set higher than the original FPS of the video.

    Args:
      video_path: A video file.
      fps: Use specific fps for video reading. (optional)

    Returns:
      A `np.array` with the shape of [seq_len, height, width, 3]
    """
    assert os.path.exists(video_path), f"{video_path} does not exist!"
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        print("stdout:", e.stdout.decode("utf8"))
        print("stderr:", e.stderr.decode("utf8"))
        raise e
    video_info = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    width = int(video_info["width"])
    height = int(video_info["height"])
    stream = ffmpeg.input(video_path)
    if fps:
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="up")
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24")
    out, _ = ffmpeg.run(stream, capture_stdout=True)
    out = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return out.copy()


def ffmpeg_video_write(data, video_path, fps=25):
    """Video writer based on FFMPEG.

    Args:
        data: A `np.array` with the shape of [seq_len, height, width, 3]
        video_path: A video file.
        fps: Use specific fps for video writing. (optional)
    """
    assert len(data.shape) == 4, f"input shape is not valid! Got {data.shape}!"
    _, height, width, _ = data.shape
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = (
        ffmpeg.input(
            "pipe:",
            framerate=fps,
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
        )
        .output(video_path, pix_fmt="yuv420p")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in data:
        writer.stdin.write(frame.astype(np.uint8).tobytes())
    writer.stdin.close()
