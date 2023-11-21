import math
import numpy as np
import cv2
import time

def object_detect():
    cap = cv2.VideoCapture('MVI_3059 - Trim.MP4')

    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=75)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    kernel2 = np.ones((50, 50), np.uint8)

    prv_center_point_arry = []

    tracking_object = {}
    track_id = 0
    objects_bbs_ids = {}

    line2_y = 500
    line1_y = 850

    frame_count = 0  # Number of frames to track an object
    total_speed = 0
    start_time = time.time()
    conversion_factor = 0.0000003
    frame_rate = 30
    object_speeds = {}  # Dictionary to store speeds for each object
    speed_list = {}
    object_start_times = {}  # Dictionary to store start times for objects



    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=0.9, fy=0.9)
        bilateral_frame = cv2.bilateralFilter(frame, 10, 180, 180)
        roi = bilateral_frame[0:1200, 1000:2000]

        mask = background_subtractor.apply(roi)
        _, detected_object = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        detected_object = cv2.morphologyEx(detected_object, cv2.MORPH_OPEN, kernel1)
        detected_object = cv2.morphologyEx(detected_object, cv2.MORPH_CLOSE, kernel2)
        contours, _ = cv2.findContours(detected_object, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cur_center_point_arry = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 15000:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = int((x + x + w) // 2)
                cy = int((y + y + h) // 2)
                cur_center_point_arry.append([cx, cy])
                cv2.circle(roi, (cx, cy), 5, (0, 0, 255), -1)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)




        for object_id, pt2 in tracking_object.items():
            cv2.circle(roi, pt2['center'], 5, (0, 0, 255), -1)
            cv2.putText(roi, str(object_id), (pt2['center'][0], pt2['center'][1] - 7), 0, 1, (0, 0, 255), 2)

            # Calculate the distance in pixels
            if 'last_position' in pt2:
                # Calculate the distance in pixels
                distance_pixels = math.hypot(pt2['center'][0] - pt2['last_position'][0],pt2['center'][1] - pt2['last_position'][1])
                distance_meters = distance_pixels * conversion_factor

                current_time1 = time.perf_counter()
                current_time2 = time.perf_counter()

                # Check if the object crosses line 1
                if pt2['last_position'][1] < line1_y > pt2['center'][1] and object_id not in object_start_times:
                    object_start_times[object_id] = current_time1
                    # print("current_time1",current_time1)

                # Check if the object crosses line 2
                if pt2['last_position'][1] < line2_y > pt2['center'][1] and object_id in object_start_times:
                    start_time_obj = current_time2
                    elapsed_time_obj = start_time_obj - current_time1
                    # print("crrent2", current_time2)
                    # print(elapsed_time_obj)

                    if elapsed_time_obj > 0:  # Ensure elapsed time is positive to avoid division by zero
                        speed_meter = distance_meters / elapsed_time_obj
                        speed_km = speed_meter * 3.6  # Convert speed to km/h

                        object_speeds[object_id] = speed_km


        tracking_object_copy = tracking_object.copy()

        for object_id, pt2 in tracking_object_copy.items():
            object_exist = False
            for pt in cur_center_point_arry:
                distance = math.hypot(pt2['center'][0] - pt[0], pt2['center'][1] - pt[1])
                if distance < 30 and object_id not in objects_bbs_ids:
                    tracking_object[object_id]['last_position'] = tracking_object[object_id]['center']
                    tracking_object[object_id]['center'] = pt
                    object_exist = True
                    cur_center_point_arry.remove(pt)
                    continue

            if not object_exist:
                if object_id in object_speeds:
                    print(f"Object {object_id} Mean Speed: {int(object_speeds[object_id])} km/h")
                else:
                    print(f"Object {object_id} not crossing line 2.")

                    objects_bbs_ids[object_id] = frame_count  # Store frame count when the object is first detected
                tracking_object.pop(object_id)


        for pt in cur_center_point_arry:
            # Check if the new object is close to an existing one
            close_to_existing = False
            for pt2 in tracking_object.values():
                distance = math.hypot(pt2['center'][0] - pt[0], pt2['center'][1] - pt[1])
                if distance < 30:
                    close_to_existing = True
                    break

            if not close_to_existing:
                tracking_object = {track_id: {'center': pt, 'last_position': pt}}
                track_id += 1


        #line 1
        cv2.line(roi, (0, 500), (1200, 500), (0, 0, 255), 2)

        #line 2
        cv2.line(roi, (0, 850), (1200, 850), (0, 0, 255), 2)

        cv2.imshow('roi', roi)
        cv2.imshow('detected_object', detected_object)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break









    cap.release()
    cv2.destroyAllWindows()

    return 0


def speed_detect():

    return 0

def main():
    object_detect()


    return 0

if __name__ == "__main__":
    main()