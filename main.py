import cv2
import numpy as np


class Processor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.pedestrian_zones = []
        self.pedestrian_zone_ids = {}
        self.count_lines = []
        self.current_zone = []
        self.current_line = []
        self.drawing = False
        self.mode = "zone"
        self.VEHICLE_COLOR = (255, 0, 0)
        self.PEDESTRIAN_COLOR = (0, 255, 0)
        self.LINE_COLOR = (0, 255, 255)
        self.VIOLATION_COLOR = (0, 0, 255)
        self.SIDEWALK_COLOR = (255, 0, 255)
        self.minArea = 1000
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.vehicleArea = 10000
        self.count_frequency = 20
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=1200, dist2Threshold=1500, detectShadows=True
        )

        self.object_positions = {}
        self.frame_time = 1 / self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = 0
        self.vehicle_count = 0
        self.pedestrian_count = 0
        _, self.first_frame = self.cap.read()

        self.sidewalk_zones = []

    def check_line_crossing(self, center_point, min_distance=20):
        for line in self.count_lines:
            p1, p2 = line
            x, y = center_point
            x1, y1 = p1
            x2, y2 = p2
            line_vector = (x2 - x1, y2 - y1)

            point_vector = (x - x1, y - y1)

            line_length = np.sqrt(line_vector[0] ** 2 + line_vector[1] ** 2)

            if line_length == 0:
                return False

            t = max(
                0,
                min(
                    1,
                    (
                        point_vector[0] * line_vector[0]
                        + point_vector[1] * line_vector[1]
                    )
                    / (line_length * line_length),
                ),
            )

            projection_x = x1 + t * line_vector[0]
            projection_y = y1 + t * line_vector[1]
            distance = np.sqrt((x - projection_x) ** 2 + (y - projection_y) ** 2)

            if (
                distance < min_distance
                and min(x1, x2) <= projection_x <= max(x1, x2)
                and min(y1, y2) <= projection_y <= max(y1, y2)
            ):
                return True

        return False

    def draw_existing_zones_and_lines(self, frame):
        for zone_id, zone in self.pedestrian_zone_ids.items():
            cv2.polylines(frame, [zone], True, self.PEDESTRIAN_COLOR, 2)
            M = cv2.moments(zone)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    frame,
                    f"ID:{zone_id}",
                    (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    self.PEDESTRIAN_COLOR,
                    2,
                )

        for zone in self.sidewalk_zones:
            cv2.polylines(frame, [zone], True, self.SIDEWALK_COLOR, 2)

        for line in self.count_lines:
            cv2.line(frame, line[0], line[1], self.LINE_COLOR, 2)

        if self.mode in ["zone", "sidewalk"] and len(self.current_zone) > 1:
            points = np.array(self.current_zone)
            color = (
                self.PEDESTRIAN_COLOR if self.mode == "zone" else self.SIDEWALK_COLOR
            )
            cv2.polylines(frame, [points], False, color, 2)
        elif self.mode == "line" and len(self.current_line) == 1:
            cv2.line(
                frame, self.current_line[0], self.current_line[0], self.LINE_COLOR, 2
            )

    def draw_text(self, frame, text, position, color=(255, 255, 255)):
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

    def mouse_callback(self, event, x, y, *_):
        temp_frame = self.first_frame.copy()
        self.draw_existing_zones_and_lines(temp_frame)

        if self.mode in ["zone", "sidewalk"]:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.current_zone.append((x, y))

            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                if len(self.current_zone) > 0:
                    color = (
                        self.PEDESTRIAN_COLOR
                        if self.mode == "zone"
                        else self.SIDEWALK_COLOR
                    )
                    cv2.line(
                        temp_frame,
                        self.current_zone[-1],
                        (x, y),
                        color,
                        2,
                    )
            cv2.imshow("Setup", temp_frame)

        elif self.mode == "line":
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.current_line) == 0:
                    self.current_line.append((x, y))
                elif len(self.current_line) == 1:
                    self.current_line.append((x, y))
                    self.count_lines.append(tuple(self.current_line))
                    self.current_line = []
            cv2.imshow("Setup", temp_frame)

    def setup(self):
        cv2.namedWindow("Setup")
        cv2.setMouseCallback("Setup", self.mouse_callback)

        print("Setup Mode:")
        print("- For pedestrian zones: Left click to set points.")
        print("- For counting lines: Draw line with two clicks.")
        print("- 'n': Start new zone (in zone or sidewalk mode).")
        print("- 'm': Change mode (Pedestrian Zone -> Sidewalk -> Counting Line).")
        print("- 'q': Finish setup and continue.")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if self.mode in ["zone", "sidewalk"] and key == ord("n"):
                if len(self.current_zone) > 2:
                    if self.mode == "zone":
                        zone_array = np.array(self.current_zone)
                        self.pedestrian_zones.append(zone_array)
                        zone_id = len(self.pedestrian_zones)
                        self.pedestrian_zone_ids[zone_id] = zone_array
                    elif self.mode == "sidewalk":
                        self.sidewalk_zones.append(np.array(self.current_zone))
                self.current_zone = []
                temp_frame = self.first_frame.copy()
                self.draw_existing_zones_and_lines(temp_frame)
                cv2.imshow("Setup", temp_frame)
                self.drawing = False

            elif key == ord("m"):
                if self.mode == "zone":
                    if len(self.current_zone) > 2:
                        self.pedestrian_zones.append(np.array(self.current_zone))
                    self.current_zone = []
                    self.mode = "sidewalk"
                    print("Switched to sidewalk zone mode.")
                elif self.mode == "sidewalk":
                    if len(self.current_zone) > 2:
                        self.sidewalk_zones.append(np.array(self.current_zone))
                    self.current_zone = []
                    self.mode = "line"
                    print("Switched to counting line mode.")
                elif self.mode == "line":
                    self.mode = "zone"
                    print("Switched to pedestrian zone mode.")

            elif key == ord("q"):
                if self.mode in ["zone", "sidewalk"] and len(self.current_zone) > 2:
                    if self.mode == "zone":
                        self.pedestrian_zones.append(np.array(self.current_zone))
                    elif self.mode == "sidewalk":
                        self.sidewalk_zones.append(np.array(self.current_zone))
                break

        cv2.destroyWindow("Setup")

    def determine_object_type(self, area, center, pedestrians_in_zones):
        color, label, objectType = None, None, None
        if area > self.vehicleArea:
            violation = False
            violation_zones = []

            for zone_id, zone in self.pedestrian_zone_ids.items():
                if cv2.pointPolygonTest(zone, center, False) >= 0:
                    if pedestrians_in_zones[zone_id]:
                        violation = True
                        violation_zones.append(zone_id)

            if violation:
                color = self.VIOLATION_COLOR
                label = (
                    f"Vehicle Violation (Zone {','.join(map(str, violation_zones))})"
                )
                objectType = "Vehicle"
            else:
                color = self.VEHICLE_COLOR
                label = "Vehicle"
                objectType = "Vehicle"

        elif area < self.vehicleArea:
            if any(
                cv2.pointPolygonTest(zone, center, False) >= 0
                for zone in self.pedestrian_zone_ids.values()
            ):
                color = self.PEDESTRIAN_COLOR
                label = "Pedestrian"
                objectType = "Pedestrian"
            elif any(
                cv2.pointPolygonTest(zone, center, False) >= 0
                for zone in self.sidewalk_zones
            ):
                color = self.PEDESTRIAN_COLOR
                label = "Pedestrian"
                objectType = "Pedestrian"
            else:
                color = self.VIOLATION_COLOR
                label = "Violation"
                objectType = "Pedestrian"
        else:
            return None, None, None
        return color, label, objectType

    def count_object(self, center, object_type):
        if object_type == "Pedestrian" and self.frame_count % self.count_frequency == 0:
            if self.check_line_crossing(center):
                self.pedestrian_count += 1
        elif (
            object_type == "Vehicle"
            and self.frame_count % (self.count_frequency / 4) == 0
        ):
            if self.check_line_crossing(center, 30):
                self.vehicle_count += 1

    def is_point_in_zones(self, point, zone_type="pedestrian"):
        if zone_type == "pedestrian":
            zones = self.pedestrian_zones
        elif zone_type == "sidewalk":
            zones = self.sidewalk_zones

        for zone in zones:
            if cv2.pointPolygonTest(zone, point, False) >= 0:
                return True
        return False

    def calculate_speed(self, obj_id, current_position):
        if obj_id in self.object_positions:
            prev_position = self.object_positions[obj_id]
            distance = np.sqrt(
                (current_position[0] - prev_position[0]) ** 2
                + (current_position[1] - prev_position[1]) ** 2
            )
            speed = distance / self.frame_time
        else:
            speed = 0

        self.object_positions[obj_id] = current_position
        return speed

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        thresh = self.bg_subtractor.apply(gray)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        thresh = cv2.dilate(thresh, self.kernel, iterations=3)
        thresh = cv2.erode(thresh, self.kernel, iterations=1)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.frame_count += 1
        objectType = None
        pedestrians_in_zones = {
            zone_id: False for zone_id in self.pedestrian_zone_ids.keys()
        }
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.minArea:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            if area < self.vehicleArea:
                for zone_id, zone in self.pedestrian_zone_ids.items():
                    if cv2.pointPolygonTest(zone, center, False) >= 0:
                        pedestrians_in_zones[zone_id] = True

        obj_id = 0

        self.draw_text(frame, f"Vehicles: {self.vehicle_count}", (10, 30))
        self.draw_text(frame, f"Pedestrians: {self.pedestrian_count}", (10, 70))

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            if area < self.minArea:
                continue

            center = (x + w // 2, y + h // 2)

            color, label, objectType = self.determine_object_type(
                area, center, pedestrians_in_zones
            )
            
            if label is not None:
                self.count_object(center, objectType)
                speed = self.calculate_speed(obj_id, center)
                label += f" {speed:.2f} px/s"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                obj_id += 1
        return frame, thresh

    def run(self):
        self.setup()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.draw_existing_zones_and_lines(frame)
            processed_frame, _ = self.process_frame(frame)

            cv2.imshow("Crossroad Monitoring", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = Processor("Test.mp4")
    monitor.run()