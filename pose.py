#!/usr/bin/env python3

import cv2
import mediapipe as mp
from pathlib import Path
import argparse
import numpy as np
import json

parser = argparse.ArgumentParser(
    description='Extract human 3D poses from videos using BlazePose model.')

parser.add_argument('input_video', type=str, help='input video')
parser.add_argument('-o', '--output', type=str, action='store', required=False,
                    help='output frames directory (will be created if does not exist)')
parser.add_argument('-v', type=bool, required=False, const=True,
                    nargs='?', default=False, help='show verbose output')
parser.add_argument('-i', '--images', type=bool, required=False, const=True,
                    nargs='?', default=False, help='output image frames')
parser.add_argument('-m', '--mask', type=bool, required=False, const=True,
                    nargs='?', default=False, help='output segmentation mask')

args = parser.parse_args()

video_path = Path(args.input_video)
video_name = video_path.stem
verbose = args.v

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BG_COLOR = (0, 0, 0)  # black
FG_COLOR = (255, 255, 255)  # white

cap = cv2.VideoCapture(str(video_path))
frame = 0
max_frames = 5000
res = ''

with mp_pose.Pose(
        enable_segmentation=args.mask,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.6) as pose:

    while cap.isOpened():
        frame += 1

        if frame > max_frames:
            break

        success, img = cap.read()

        if not success:
            break

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # If no output directory specified
        # Draw the pose annotation on the image adn display preview to user
        if not args.output:
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            cv2.imshow('MediaPipe Pose', img)

            if cv2.waitKey(5) & 0xFF == 27:
                break

            frame += 1
            continue

        output_dir = Path(args.output)
        # Output images
        if args.output and args.images:
            output_dir_image = output_dir / "img"
            img_path = output_dir_image / (str(frame) + ".png")
            output_dir_image.mkdir(parents=True, exist_ok=True)
            
            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), image)

        if not hasattr(results.pose_world_landmarks, 'landmark'):
            print(f'[-] No ladmark found on frame {frame}.')
            continue

        if (verbose):
            print('Processing frame', frame)

        # Save calculated coordinates into files
        pose_dir = output_dir / 'pose'

        pose_dir.mkdir(parents=True, exist_ok=True)

        frame_path = pose_dir / str(f'{frame}.json')

        landmarks = results.pose_world_landmarks.landmark

        output_data = {
            "world": list(map(lambda i: [i.x, i.y, i.z], results.pose_world_landmarks.landmark)),
            "screen": list(map(lambda i: [i.x, i.y, i.z], results.pose_landmarks.landmark)),
            "resolution": [image.shape[0], image.shape[1]],
        }

        with open(frame_path, 'w') as output_file:
            output_file.write(json.dumps(
                output_data, indent=4, sort_keys=True))

        # Output segmentation mask
        if args.mask:
            output_dir_mask = output_dir / "mask"
            mask_path = output_dir_mask / (str(frame) + ".png")

            output_dir_mask.mkdir(parents=True, exist_ok=True)

            masked_image = image.copy()
            masked_image[:] = FG_COLOR

            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            condition = np.stack(
                (results.segmentation_mask,) * 3, axis=-1) > 0.1
            bw_mask = np.where(
                condition, masked_image, bg_image)

            # Smooth mask edges
            filtered = cv2.ximgproc.jointBilateralFilter(masked_image, bw_mask, d=-1, sigmaColor=25, sigmaSpace=4)

            cv2.imwrite(str(mask_path), filtered)

cap.release()
