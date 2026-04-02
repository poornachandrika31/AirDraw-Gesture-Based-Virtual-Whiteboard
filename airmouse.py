import sys
import numpy as np
import cv2 as cv
import pyautogui
from hands import HandDetector, Gesture

# Safety fail-safe for PyAutoGUI (moving mouse to corner of screen aborts)
pyautogui.FAILSAFE = True
# Turn off pause so it doesn't freeze camera stream
pyautogui.PAUSE = 0.05

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

def main():
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
    
    # On Windows, cv.CAP_DSHOW (DirectShow) gives faster camera init.
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        sys.exit(1)

    detector = HandDetector()

    cv.namedWindow("AirMouse Mode", cv.WINDOW_NORMAL)
    cv.resizeWindow("AirMouse Mode", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    # Coordinates smoothing to avoid jittering
    prev_x, prev_y = 0, 0
    smooth_factor = 0.35  

    print("AirMouse Controls:")
    print(" - Index finger up (HOVER): Move mouse freely")
    print(" - Index + Middle fingers up (DRAW): Click / Drag")
    print(" - 'q' or ESC: Quit")
    
    mouse_down = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # -1 = flip both axes (mirror + upside-down, same as airdraw.py)
        frame = cv.flip(frame, -1)  
        frame = cv.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv.INTER_LINEAR)

        # Gets data from our existing hands logic
        gesture_metadata = detector.get_gesture_metadata(frame)
        detector.draw_landmarks(frame)
        
        if 'gesture' in gesture_metadata and gesture_metadata['gesture'] is not None:
            gesture = gesture_metadata['gesture']
            tip = gesture_metadata.get('idx_fing_tip', None)
            
            if tip is not None:
                # tip is (row, col) => (y, x)
                cam_y, cam_x = tip
                
                # We amplify the movement so you don't have to reach the very edge of the camera
                # Larger margin means less hand movement needed to cross the screen.
                margin_x = 200
                margin_y = 150
                target_x = np.interp(cam_x, (margin_x, DISPLAY_WIDTH - margin_x), (0, SCREEN_WIDTH))
                target_y = np.interp(cam_y, (margin_y, DISPLAY_HEIGHT - margin_y), (0, SCREEN_HEIGHT))
                
                # Smooth the movement: lower values = smoother but slightly delayed
                smooth_factor = 0.2  
                curr_x = prev_x + (target_x - prev_x) * smooth_factor
                curr_y = prev_y + (target_y - prev_y) * smooth_factor
                
                # --- APPLY MOUSE ACTIONS ---
                try:
                    # Hovering/Moving
                    if gesture == Gesture.HOVER or gesture == Gesture.TRANSLATE or gesture == Gesture.ERASE:
                        mouse_down = False
                        pyautogui.moveTo(curr_x, curr_y)
                    
                    # Single Click
                    elif gesture == Gesture.DRAW:
                        if not mouse_down:
                            pyautogui.click()
                            mouse_down = True
                        pyautogui.moveTo(curr_x, curr_y)

                except pyautogui.FailSafeException:
                    mouse_down = False # Failsafe triggered on corner

                prev_x, prev_y = curr_x, curr_y

        cv.imshow("AirMouse Mode", frame)

        stroke = cv.waitKey(1) & 0xFF
        if stroke == ord('q') or stroke == 27:
            break

    # Clean up
    cap.release()
    cv.destroyAllWindows()
    if mouse_down:
        pyautogui.mouseUp()

if __name__ == '__main__':
    main()
