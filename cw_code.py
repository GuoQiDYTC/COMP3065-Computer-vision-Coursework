import cv2
import numpy as np
import logging
import functools
import time

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def time_it(func):
    """Wrapper: Calculates and prints the execution time of a function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.2f} seconds.")
        return result
    return wrapper

def read_video(video_path):
    """Reads the video and returns frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def sharpen_image(image):
    """Sharpening kernel is used to enhance image details"""
    # 定义一个锐化核
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    # 应用锐化核
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened_image

@time_it
def sharpen_frame(frames):
    """sharpens all the frames in the list of frames"""
    for frame in frames:
        frame = sharpen_image(frame)
    return frames
        

@time_it
def filter_motion_frames(frames, motion_threshold=0.5):
    """Removes duplicate frames based on the average magnitude of optical flow between frames"""
    filtered_frames = [frames[0]]  # Always include the first frame
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # using downsampled images to improve performance
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        small_prev_gray = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)
        
        flow = cv2.calcOpticalFlowFarneback(small_prev_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_magnitude = np.mean(magnitude)

        if mean_magnitude > motion_threshold:
            filtered_frames.append(frame)
        
        prev_gray = gray

    return filtered_frames

@time_it
def select_key_frames(frames, skip=5):
    """Select key frames by skipping frames"""
    key_frames = frames[::skip]
    return key_frames

@time_it
def detect_and_match_features(frames):
    """Detects features and matches them between frames"""
    orb = cv2.ORB_create()
    keypoints_all = []
    descriptors_all = []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for frame in frames:
        kp, des = orb.detectAndCompute(frame, None)
        keypoints_all.append(kp)
        descriptors_all.append(des)

    matches = []
    for i in range(len(descriptors_all) - 1):
        match = matcher.match(descriptors_all[i], descriptors_all[i+1])
        matches.append(sorted(match, key=lambda x: x.distance))
    return keypoints_all, matches

@time_it
def stitch(frames, keypoints_all, matches):
    """Stitches the images together"""
    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(frames)

    if status == cv2.Stitcher_OK:
        return stitched
    else:
        logging.error(f"Error in stitching: {status}")
        return None


def main(video_path, apply_sharpen=False, apply_motion_filter=False,motion_threshold = 0.8, skip_frames=10, output_name='panorama.jpg'):
    frames = read_video(video_path)
    if apply_sharpen:
        frames = sharpen_frame(frames)
    if apply_motion_filter:
        frames = filter_motion_frames(frames, motion_threshold=motion_threshold)
    frames = select_key_frames(frames, skip=skip_frames)
    keypoints_all, matches = detect_and_match_features(frames)
    panorama = stitch(frames, keypoints_all, matches)

    if panorama is not None:
        cv2.imshow('Panorama', panorama)
        cv2.imwrite(output_name, panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'with_stop.mp4'  # video path
    main(video_path, apply_sharpen=False, apply_motion_filter=True, motion_threshold = 0.8, skip_frames=10, output_name='panorama.jpg')
