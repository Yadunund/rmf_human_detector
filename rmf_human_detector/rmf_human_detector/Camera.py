#!/usr/bin/env python3
from pathlib import Path
import sys
import cv2
import depthai as dai
import blobconverter
import numpy as np
import time
import threading

class Detection:
    def __init__(self,depth_x,depth_y,depth_z,width,height):
        self.label = "person"
        self.depth_x = depth_x
        self.depth_y = depth_y
        self.depth_z = depth_z
        self.width = width
        self.height = height
'''
Spatial detection network demo.
    Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
'''
class Camera:
    def __init__(self, visualize=True):
        self.visualize = visualize
        self.detections = [] # store detected persons as Detection objects
        self.pipeline = self._create_pipeline()
        self.quit = True
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def start(self):
        print("Starting to detect humans...")
        self.quit = False

    def stop(self):
        self.quit = True

    def __del__(self):
        self.stop()
        del self.pipeline
        self.thread = None

    def get_detections(self):
        with self._lock:
            return self.detections

    def _create_pipeline(self):

        nnBlobPath = blobconverter.from_zoo(name='mobilenet-ssd', shaves=6)
        # MobilenetSSD label texts
        self.labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        self.syncNN = True

        # Create pipeline
        pipeline = dai.Pipeline()

        # Add nodes for rgb, depth, and nn
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        # Create connections
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutBoundingBoxDepthMapping = pipeline.create(dai.node.XLinkOut)
        xoutDepth = pipeline.create(dai.node.XLinkOut)

        # Create message streams
        xoutRgb.setStreamName("rgb")
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth.setStreamName("depth")

        # Set properties
        camRgb.setPreviewSize(300, 300)
        # camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Setting node configs
        stereo.initialConfig.setConfidenceThreshold(255)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

        spatialDetectionNetwork.setBlobPath(nnBlobPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.7)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)
        spatialDetectionNetwork.setNumInferenceThreads(2)

        # Link nodes
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        else:
            camRgb.preview.link(xoutRgb.input)

        spatialDetectionNetwork.out.link(xoutNN.input) # detections
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

        return pipeline

    def _run(self):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            while True:
                if self.quit:
                    time.sleep(1)
                    continue

                inDet = detectionNNQueue.get()
                if inDet is None:
                    continue

                depth = depthQueue.get()
                depthFrame = depth.getFrame() # depthFrame values are in millimeters

                detections = inDet.detections
                roiDatas = xoutBoundingBoxDepthMapping.get().getConfigData()

                num_detections = len(detections)
                num_rois = len(roiDatas)
                # print(f"num_detections: {num_detections} num_rois: {num_rois}")
                if (num_detections != num_rois):
                    # print(f"    num_detections and num_rois do not match. Skipping")
                    continue
                # todo track objects and update on when there is significant change in position

                if (self.visualize):
                    frame = previewQueue.get().getCvFrame()
                    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depthFrameColor = cv2.equalizeHist(depthFrameColor)
                    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                new_detections = []
                for i in range(len(detections)):
                    detection = detections[i]
                    roi = roiDatas[i].roi.denormalize(depthFrame.shape[1], depthFrame.shape[0])
                    if detection.label >= len(self.labelMap):
                        # print(f"Detected unknown label: {detection.label}")
                        continue
                    label = self.labelMap[detection.label]
                    # print(f"Detected {label}")

                    if (self.visualize):
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)
                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), 255, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                        frame_width = frame.shape[1]
                        frame_height = frame.shape[0]
                        x1 = int(detection.xmin * frame_width)
                        x2 = int(detection.xmax * frame_width)
                        y1 = int(detection.ymin * frame_height)
                        y2 = int(detection.ymax * frame_height)

                        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

                    if label != "person":
                        # print(f"    Not adding detection {label} to detections list")
                        continue

                    # todo(YV): Get the correct width and height
                    width = roi.size().width / 1000.0
                    height = roi.size().height / 1000.0
                    depth_x = detection.spatialCoordinates.x / 1000.0
                    depth_y = detection.spatialCoordinates.y / 1000.0
                    depth_z = detection.spatialCoordinates.z / 1000.0
                    new_detections.append(Detection(depth_x,depth_y,depth_z,0.6,1.8))
                    # print(f"Detected person of height {height : .2f} and width {width : .2f}")
                
                with self._lock:
                    self.detections = new_detections

                if (self.visualize):
                    cv2.imshow("depth", depthFrameColor)
                    cv2.imshow("preview", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

if __name__ == '__main__':
    c = Camera()
    c.start()