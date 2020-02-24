#!/usr/bin/env python
# -*- coding: utf-8 -*-

from uvctypes import *
import time
import cv2
import numpy as np
try:
  from queue import Queue
except ImportError:
  from Queue import Queue
import platform

BUF_SIZE = 2
q = Queue(BUF_SIZE)

def py_frame_callback(frame, userptr):

  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
  data = np.frombuffer(
    array_pointer.contents, dtype=np.dtype(np.uint16)
  ).reshape(
    frame.contents.height, frame.contents.width
  ) # no copy

  # data = np.fromiter(
  #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
  # ).reshape(
  #   frame.contents.height, frame.contents.width, 2
  # ) # copy

  if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
    return

  if not q.full():
    q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
  # GENE MODIFIED, NOT SURE WHY DATA IS INCORRECT, MANUAL CORRECTION VIA SCALE / OFFSET
  #return (val - 27315) / 100.0
  return val / 100 * 4 - 27315 / 100 - 25

def raw_to_8bit(data):
  cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
  np.right_shift(data, 8, data)
  return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

def display_temperature(img, val_k, loc, color):
  val = ktof(val_k)
  cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

def main():
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devh = POINTER(uvc_device_handle)()
  ctrl = uvc_stream_ctrl()

  res = libuvc.uvc_init(byref(ctx), 0)
  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
      print("uvc_find_device error")
      exit(1)

    try:
      res = libuvc.uvc_open(dev, byref(devh))
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")

      print_device_info(devh)
      print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
      )

      res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)

      try:
        while True:
          data = q.get(True, 500)
          if data is None:
            break
          data = cv2.resize(data[:,:], (640, 480))
          minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
          img = raw_to_8bit(data)

          # Setup SimpleBlobDetector parameters.
          params = cv2.SimpleBlobDetector_Params()

          # DEBUGGING
          # print("minVal: ", minVal)
          # print("maxVal: ", maxVal)

          # Compute histogram of image
          hist = cv2.calcHist([img], [0], None, [256], [0, 255])

          # Extract median temperature in image
          medianTemp8Bit = np.argmax(hist)
          medianTempRaw = minVal + (maxVal - minVal) * medianTemp8Bit / 255
          medianTempF = 1.8 * ( medianTempRaw / 100 * 4 - 27315 / 100 - 25 ) + 32.0

          # DEBUGGING
          # print("medianTemp8BitIdx: ",  medianTemp8Bit)
          # print("medianTempF: ",        medianTempF)

          # Set blob detector thresholds to 20-30C above ambient
          minThresF = medianTempF + 1.8*10
          maxThresF = medianTempF + 1.8*30
          minThresRaw = ( ( ( (minThresF - 32.0) / 1.8 ) + 27315.0 / 100.0 + 25.0 ) * 100.0 / 4.0 )
          maxThresRaw = ( ( ( (maxThresF - 32.0) / 1.8 ) + 27315.0 / 100.0 + 25.0 ) * 100.0 / 4.0 )
          params.minThreshold = min( 1.0, max( 0.0, (minThresRaw - minVal) / (maxVal - minVal) ) ) * 255.0
          params.maxThreshold = min( 1.0, max( 0.0, (maxThresRaw - minVal) / (maxVal - minVal) ) ) * 255.0
          
          # Find min and max temperature in image
          minTemp8Bit = img.min()
          maxTemp8Bit = img.max()
          minTempF = ( 1.8 * ( minVal / 100 * 4 - 27315 / 100 - 25 ) + 32 )
          maxTempF = ( 1.8 * ( maxVal / 100 * 4 - 27315 / 100 - 25 ) + 32 )

          print("Min/Max     Temp - Blob Limit 8 Bit:     {0:3.0f}/255      {1:3.0f}/255".format(params.minThreshold, params.maxThreshold))
          print("Min/Max     Temp - Blob Limit F:         {0:3.0f}F         {1:3.0f}F".format(minThresF, maxThresF) )
          print("Min/Max/Med Temp - Image Measured 8 Bit: {0:3.0f}/255      {1:3.0f}/255      {2:3.0f}/255".format(minTemp8Bit, maxTemp8Bit, medianTemp8Bit) )
          print("Min/Max/Med Temp - Image Measured Lim:   {0:3.0f}F         {1:3.0f}F         {2:3.0f}F".format(minTempF, maxTempF, medianTempF) )
          
          # Filter by Area.
          params.filterByArea = True
          params.minArea = 40*20
          params.maxArea = 640*480
          params.filterByCircularity = False
          params.minCircularity = 0.5
          params.filterByColor = False
          params.filterByConvexity = False
          params.minConvexity = 0.50
          params.filterByInertia = False
          params.minInertiaRatio = 0.1

          # Create a detector with the parameters
          ver = (cv2.__version__).split('.')
          if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
          else: 
            detector = cv2.SimpleBlobDetector_create(params)

          # Detect blobs.
          keypoints = detector.detect(img)

          # Draw detected blobs as red circles.
          # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
          # the size of the circle corresponds to the size of blob

          img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

          display_temperature(img_with_keypoints, minVal, minLoc, (255, 0, 0))
          display_temperature(img_with_keypoints, maxVal, maxLoc, (0, 0, 255))

          # Show blobs
          cv2.imshow("Keypoints", img_with_keypoints)

          # cv2.imshow('Lepton Radiometry', img)
          cv2.waitKey(1)

        cv2.destroyAllWindows()
      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(dev)
  finally:
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
  main()
