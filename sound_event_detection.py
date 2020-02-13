import os
import sys
import argparse
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], 'audioset_tagging_cnn/utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'audioset_tagging_cnn/pytorch'))
from inference import sound_event_detection
import cv2


def main(args):
    # Sound event detection
    (framewise_output, labels) = sound_event_detection(args)
    """framewise_output: (frames_num, classes_num)"""
    
    # Add detected results text to video
    add_text_to_video(framewise_output, labels, args.video_path, args.out_video_path)


def add_text_to_video(framewise_output, labels, video_path, out_video_path):
    
    topk = 5    # Number of sound classes to show
    sed_frames_per_second = 100

    # Paths
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

    tmp_video_path = '_tmp/tmp.avi'
    os.makedirs('_tmp', exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('frame_width: {}, frame_height: {}, fps: {}'.format(
        frame_width, frame_height, fps))

    assert fps > 29 and fps <= 30
    assert frame_width == 1920
    assert frame_height == 1080

    sed_frames_per_video_frame = sed_frames_per_second / float(fps)

    out = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 
        fps, (frame_width, frame_height))

    # For each frame select the top classes
    sorted_indexes = np.argsort(framewise_output, axis=-1)[:, -1 : -topk - 1 : -1]
    """(frames_num, topk)"""

    n = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # End of video
        if frame is None:
            break

        cv2.rectangle(frame, (0, 0), (900, 450), (255, 255, 255), -1)

        for k in range(0, topk):
            # Add text to frames
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (20, 90 + k * 80)
            fontScale              = 2
            fontColor              = (0,0,255)
            lineType               = 3
 
            m = int(n * sed_frames_per_video_frame)
            
            text = '{}: {:.3f}'.format(
                cut_words(labels[sorted_indexes[m, k]]), 
                framewise_output[m, sorted_indexes[m, k]])

            cv2.putText(frame, text, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

        # Write frame to video
        out.write(frame)

        n += 1

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    os.system('ffmpeg -loglevel panic -y -i {} "{}"'.format(tmp_video_path, out_video_path))
    print('Write silent video with text to {}'.format(out_video_path))


def cut_words(lb, max_len=20):
    """Cut label to max_len.
    """
    words = lb.split(', ')
    new_lb = ''
    for word in words:
        if len(new_lb + ', ' + word) > max_len:
            break
        new_lb += ', ' + word
    new_lb = new_lb[2 :]

    if len(new_lb) == 0:
        new_lb = words[0]
    
    return new_lb


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser.')    
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000) 
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--out_video_path', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()

    main(args)