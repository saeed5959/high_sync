# pylint: disable=W0718
"""
This module is responsible for processing images, particularly for face-related tasks.
It uses various libraries such as OpenCV, NumPy, and InsightFace to perform tasks like
face detection, augmentation, and mask rendering. The ImageProcessor class encapsulates
the functionality for these operations.
"""
import os
from typing import List

import cv2
import mediapipe as mp
import numpy as np
import torch
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps
from torchvision import transforms

from ..utils.util import (blur_mask, get_landmark_overframes, get_mask,
                          get_union_face_mask, get_union_lip_mask)

MEAN = 0.5
STD = 0.5

class ImageProcessor:
    """
    ImageProcessor is a class responsible for processing images, particularly for face-related tasks.
    It takes in an image and performs various operations such as augmentation, face detection,
    face embedding extraction, and rendering a face mask. The processed images are then used for
    further analysis or recognition purposes.

    Attributes:
        img_size (int): The size of the image to be processed.
        face_analysis_model_path (str): The path to the face analysis model.

    Methods:
        preprocess(source_image_path, cache_dir):
            Preprocesses the input image by performing augmentation, face detection,
            face embedding extraction, and rendering a face mask.

        close():
            Closes the ImageProcessor and releases any resources being used.

        _augmentation(images, transform, state=None):
            Applies image augmentation to the input images using the given transform and state.

        __enter__():
            Enters a runtime context and returns the ImageProcessor object.

        __exit__(_exc_type, _exc_val, _exc_tb):
            Exits a runtime context and handles any exceptions that occurred during the processing.
    """
    def __init__(self, img_size, face_analysis_model_path) -> None:
        self.img_size = img_size

        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([MEAN], [STD]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )

        self.attn_transform_64 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 8, self.img_size[0] // 8)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_32 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 16, self.img_size[0] // 16)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_16 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 32, self.img_size[0] // 32)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_8 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 64, self.img_size[0] // 64)),
                transforms.ToTensor(),
            ]
        )

        self.face_analysis = FaceAnalysis(
            name="",
            root=face_analysis_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    def preprocess(self, frame):
        """
        Apply preprocessing to the source image to prepare for face analysis.

        Parameters:
            source_image_path (str): The path to the source image.
            cache_dir (str): The directory to cache intermediate results.

        Returns:
            None
        """
        frame_h, frame_w, _ = frame.shape
        # 2.1 detect face
        faces = self.face_analysis.get(frame)
        if not faces:
            print("No faces detected in the image. Using the entire image as the face region.")
            bbox = (0,0,0,0)
            face_emb = np.zeros(512)
            frame_crop = frame
        else:
            # Sort faces by size and select the largest one
            face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]), reverse=True)[0]
            face_emb = face["embedding"]
            w, h = face["bbox"][2] - face["bbox"][0], face["bbox"][3] - face["bbox"][1]
            bbox = (int(max(face["bbox"][0] - 0.1*w, 0)), int(face["bbox"][1]),
                                 int(min(face["bbox"][2] + 0.1*w, frame_w)), int(min(face["bbox"][3] + 0.1*h, frame_h)))
            frame_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        cv2.imwrite("./x.jpg", frame_crop)
        frame_crop = cv2.resize(frame_crop, (512,512))

        pixel_values_ref_img = self._augmentation(Image.fromarray(frame_crop).convert("RGB"), self.pixel_transform)

        return pixel_values_ref_img, face_emb, bbox, frame_crop
    

    def preprocess_frames(self, frames):
        """
        Apply preprocessing to the 14 source images to prepare for face analysis.

        Parameters:
            source_image_path (str): The path to the source image.
            cache_dir (str): The directory to cache intermediate results.

        Returns:
            None
        """
        def find_max_h(bbox_list):
    
            h_max = 0
            for bbox in bbox_list:
                h = bbox[3]
                if h>h_max:
                    h_max = h

            return h_max

        pixel_values_ref_img_list, bbox_list, bbox_final_list, frame_crop_list = [], [], [], []

        face_emb = np.zeros(512)

        for frame in frames:

            # 2.1 detect face
            faces = self.face_analysis.get(frame)
            if not faces:
                print("No faces detected in the image. Using the entire image as the face region.")
                bbox_list.append((0,0,0,0))
                
            else:
                # Sort faces by size and select the largest one
                face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]), reverse=True)[0]
                face_emb = face["embedding"]
                bbox_list.append(face["bbox"])
            
        h_max = find_max_h(bbox_list)

        for frame, bbox in zip(frames, bbox_list):
            
            frame_h, frame_w, _ = frame.shape

            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            if h_max < 1.4*h:
                h = h_max
        
            bbox_final = (int(max(bbox[0] - 0.1*w, 0)), int(bbox[1]),
                                 int(min(bbox[2] + 0.1*w, frame_w)), int(min(bbox[1] + 1.1*h, frame_h)))
            bbox_final_list.append(bbox_final)

            frame_crop = frame[bbox_final[1]:bbox_final[3], bbox_final[0]:bbox_final[2]]
            frame_crop_list.append(cv2.resize(frame_crop, (512,512)))

            pixel_values_ref_img_list.append(self._augmentation(Image.fromarray(frame_crop).convert("RGB"), self.pixel_transform).unsqueeze(0))

        return pixel_values_ref_img_list, face_emb, bbox_list, frame_crop_list
    

    def close(self):
        """
        Closes the ImageProcessor and releases any resources held by the FaceAnalysis instance.

        Args:
            self: The ImageProcessor instance.

        Returns:
            None.
        """
        for _, model in self.face_analysis.models.items():
            if hasattr(model, "Dispose"):
                model.Dispose()

    def _augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()


class ImageProcessorForDataProcessing():
    """
    ImageProcessor is a class responsible for processing images, particularly for face-related tasks.
    It takes in an image and performs various operations such as augmentation, face detection,
    face embedding extraction, and rendering a face mask. The processed images are then used for
    further analysis or recognition purposes.

    Attributes:
        img_size (int): The size of the image to be processed.
        face_analysis_model_path (str): The path to the face analysis model.

    Methods:
        preprocess(source_image_path, cache_dir):
            Preprocesses the input image by performing augmentation, face detection,
            face embedding extraction, and rendering a face mask.

        close():
            Closes the ImageProcessor and releases any resources being used.

        _augmentation(images, transform, state=None):
            Applies image augmentation to the input images using the given transform and state.

        __enter__():
            Enters a runtime context and returns the ImageProcessor object.

        __exit__(_exc_type, _exc_val, _exc_tb):
            Exits a runtime context and handles any exceptions that occurred during the processing.
    """
    def __init__(self, face_analysis_model_path, landmark_model_path, step) -> None:
        if step == 2:
            self.face_analysis = FaceAnalysis(
                name="",
                root=face_analysis_model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
            self.landmarker = None
        else:
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            # Create a face landmarker instance with the video mode:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=landmark_model_path),
                running_mode=VisionRunningMode.IMAGE,
            )
            self.landmarker = None #FaceLandmarker.create_from_options(options)
            self.face_analysis = None

    def preprocess(self, source_image_path: str):
        """
        Apply preprocessing to the source image to prepare for face analysis.

        Parameters:
            source_image_path (str): The path to the source image.
            cache_dir (str): The directory to cache intermediate results.

        Returns:
            None
        """
        # 1. get face embdeding
        face_mask, face_emb, sep_pose_mask, sep_face_mask, sep_lip_mask = None, None, None, None, None
        if self.face_analysis:
            frame = sorted(os.listdir(source_image_path))[0]
            source_image = Image.open(
                os.path.join(source_image_path, frame))
            ref_image_pil = source_image.convert("RGB")

            padding = 250
            ref_image_pil = ImageOps.expand(ref_image_pil, border=padding, fill=0)

            # 2.1 detect face
            faces = self.face_analysis.get(cv2.cvtColor(
                np.array(ref_image_pil.copy()), cv2.COLOR_RGB2BGR))

            # use max size face
            face = sorted(faces, key=lambda x: (
                x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
            # 2.2 face embedding
            face_emb = face["embedding"]
            if face_emb is None:
                print(f"error in image embedding : {source_image_path}")
                raise

        if self.landmarker:
            # 3.1 get landmark
            landmarks, height, width = get_landmark_overframes(
                self.landmarker, source_image_path)
            assert len(landmarks) == len(os.listdir(source_image_path))

            # 3 render face and lip mask
            face_mask = get_union_face_mask(landmarks, height, width)
            lip_mask = get_union_lip_mask(landmarks, height, width)

            # 4 gaussian blur
            blur_face_mask = blur_mask(face_mask, (64, 64), (51, 51))
            blur_lip_mask = blur_mask(lip_mask, (64, 64), (31, 31))

            # 5 seperate mask
            sep_face_mask = cv2.subtract(blur_face_mask, blur_lip_mask)
            sep_pose_mask = 255.0 - blur_face_mask
            sep_lip_mask = blur_lip_mask

        return face_mask, face_emb, sep_pose_mask, sep_face_mask, sep_lip_mask

    def close(self):
        """
        Closes the ImageProcessor and releases any resources held by the FaceAnalysis instance.

        Args:
            self: The ImageProcessor instance.

        Returns:
            None.
        """
        for _, model in self.face_analysis.models.items():
            if hasattr(model, "Dispose"):
                model.Dispose()

    def _augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
