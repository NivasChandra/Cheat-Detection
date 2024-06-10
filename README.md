This project is a real-time head pose detection system designed to monitor and detect potential cheating during online examinations or any activity requiring constant attention to the screen. The system uses OpenCV for video capture and MediaPipe for face and pose detection, issuing warnings for specific behaviors such as looking away, multiple faces detected, or no face detected. The `is_cheating` function processes a given image to detect head pose and determine if the user is looking away, while the `capture_and_check` function captures video frames from the webcam, checks for cheating behavior, and issues warnings, terminating if the maximum number of warnings is reached. The `suppress_stderr` function suppresses TensorFlow warnings for cleaner output. The system can be configured by modifying thresholds and warning limits in the script. 