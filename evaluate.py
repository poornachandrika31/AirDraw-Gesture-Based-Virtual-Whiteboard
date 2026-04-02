import time
import numpy as np
import cv2 as cv
from hands import HandDetector, Gesture

def run_evaluation():
    # Initialize camera and detector
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = HandDetector()
    
    # Metrics containers
    fps_list = []
    inference_times = []
    jitter_data = []
    
    print("\n--- CV Canvas Evaluation Tool ---")
    print("Press 'f' to start FPS & Latency test (10 seconds)")
    print("Press 'j' to start Jitter/Stability test (Hold index finger still for 5 seconds)")
    print("Press 'g' to test Gesture Accuracy (Manual verification)")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Display current feed
        cv.putText(frame, "EVALUATION MODE", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(frame, "f: FPS/Latency | j: Jitter | g: Gesture | q: Quit", (10, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv.imshow("Evaluator", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # --- 1. FPS and Latency Test ---
        elif key == ord('f'):
            print("\n[FPS/Latency] Starting 10-second benchmark...")
            start_test = time.time()
            frames_processed = 0
            latencies = []
            
            while time.time() - start_test < 10:
                inner_ret, inner_frame = cap.read()
                if not inner_ret: break
                
                t0 = time.perf_counter()
                metadata = detector.get_gesture_metadata(inner_frame)
                t1 = time.perf_counter()
                
                latencies.append((t1 - t0) * 1000) # Convert to ms
                frames_processed += 1
                
                # Feedback on screen
                draw_frame = cv.flip(inner_frame, 1)
                cv.putText(draw_frame, f"Benchmarking... {int(10 - (time.time() - start_test))}s left", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imshow("Evaluator", draw_frame)
                cv.waitKey(1)

            avg_fps = frames_processed / 10
            avg_latency = np.mean(latencies)
            print(f"RESULTS:")
            print(f" - Average FPS: {avg_fps:.2f}")
            print(f" - Average Inference Latency: {avg_latency:.2f} ms")

        # --- 2. Jitter / Stability Test ---
        elif key == ord('j'):
            print("\n[Jitter] Hold your index finger STILL in the center...")
            time.sleep(1) # Prep time
            start_test = time.time()
            positions = []
            
            while time.time() - start_test < 5:
                inner_ret, inner_frame = cap.read()
                if not inner_ret: break
                
                landmarks = detector.detect_landmarks(inner_frame)
                if landmarks:
                    # Index finger tip is landmark 8
                    _, x, y = landmarks[8]
                    positions.append((x, y))
                
                draw_frame = cv.flip(inner_frame, 1)
                cv.putText(draw_frame, f"HOLD STILL... {int(5 - (time.time() - start_test))}s left", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv.imshow("Evaluator", draw_frame)
                cv.waitKey(1)

            if positions:
                pos_arr = np.array(positions)
                std_dev = np.std(pos_arr, axis=0)
                jitter_score = np.sqrt(np.sum(std_dev**2))
                print(f"RESULTS:")
                print(f" - Stability Score (Lower is better): {jitter_score:.4f} pixels")
                print(f" - Std Dev X: {std_dev[0]:.4f}, Y: {std_dev[1]:.4f}")
            else:
                print("No hand detected during test.")

        # --- 3. Gesture Accuracy Test ---
        elif key == ord('g'):
            gestures_to_test = [Gesture.HOVER, Gesture.DRAW, Gesture.ERASE]
            results = {g: {"correct": 0, "total": 0} for g in gestures_to_test}
            
            print("\n[Gesture Accuracy] Perform the requested gesture and tell me if it's correct.")
            
            for target in gestures_to_test:
                print(f"\nPerform: {target.value}")
                print("Press 'y' if Correct, 'n' if Incorrect")
                
                test_start = time.time()
                while True:
                    inner_ret, inner_frame = cap.read()
                    if not inner_ret: break
                    
                    data = detector.get_gesture_metadata(inner_frame)
                    detected = data.get('gesture', 'None')
                    
                    draw_frame = cv.flip(inner_frame, 1)
                    cv.putText(draw_frame, f"TARGET: {target.value}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv.putText(draw_frame, f"DETECTED: {detected.value if hasattr(detected, 'value') else detected}", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv.putText(draw_frame, "Press 'y' (Correct) or 'n' (Wrong)", (10, h-50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv.imshow("Evaluator", draw_frame)
                    
                    inner_key = cv.waitKey(1) & 0xFF
                    if inner_key == ord('y'):
                        results[target]["correct"] += 1
                        results[target]["total"] += 1
                        break
                    elif inner_key == ord('n'):
                        results[target]["total"] += 1
                        break
            
            print("\nACCURACY RESULTS:")
            for g, data in results.items():
                acc = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
                print(f" - {g.value}: {acc:.1f}%")

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    run_evaluation()
