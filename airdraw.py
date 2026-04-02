import sys
import numpy as np
import cv2 as cv
from hands import HandDetector
from canvas import Canvas, Color

def get_scaled_background(img, scale_factor, disp_w, disp_h):
    """Safely scales and centers an image onto a white whiteboard background."""
    h, w = img.shape[:2]
    base_scale = min(disp_w / w, disp_h / h)
    final_scale = base_scale * scale_factor
    
    new_w, new_h = int(w * final_scale), int(h * final_scale)
    bg = np.ones((disp_h, disp_w, 3), dtype=np.uint8) * 255
    
    if new_w <= 0 or new_h <= 0:
        return bg
        
    resized = cv.resize(img, (new_w, new_h))
    
    x_offset = (disp_w - new_w) // 2
    y_offset = (disp_h - new_h) // 2
    
    y1, y2 = max(0, y_offset), min(disp_h, y_offset + new_h)
    x1, x2 = max(0, x_offset), min(disp_w, x_offset + new_w)
    
    img_y1, img_x1 = max(0, -y_offset), max(0, -x_offset)
    img_y2, img_x2 = img_y1 + (y2 - y1), img_x1 + (x2 - x1)
    
    if y2 > y1 and x2 > x1:
        bg[y1:y2, x1:x2] = resized[img_y1:img_y2, img_x1:img_x2]
    return bg

# ── Display resolution ────────────────────────────────────────────────────────
# The webcam captures at its native resolution (usually 640×480).
# We upscale every frame to DISPLAY_WIDTH × DISPLAY_HEIGHT before rendering.
# The Canvas is also initialised at this size so button hit-detection and
# drawing coordinates are always in "display space" — no offset issues.
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720
# ─────────────────────────────────────────────────────────────────────────────


def replay(fname):
    print("replaying", fname)

    cap = cv.VideoCapture(fname)

    if not cap.isOpened():
        print(f"Error: could not open video file '{fname}'")
        return

    canvas   = Canvas(DISPLAY_HEIGHT, DISPLAY_WIDTH)
    detector = HandDetector()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret or img is None:
            break

        # Scale to display size so replay looks the same as live mode
        img = cv.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv.INTER_LINEAR)

        gesture_metadata = detector.get_gesture_metadata(img)
        img = canvas.update_and_draw(img, gesture_metadata)
        detector.draw_landmarks(img)

        cv.imshow('AirDraw Replay', img)

        key = cv.waitKey(33) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    print("replay complete", fname)


def main():
    # On Windows, cv.CAP_DSHOW (DirectShow) gives faster camera init.
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: could not open webcam. Make sure a camera is connected and not in use by another app.")
        sys.exit(1)

    cam_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)  + 0.5)
    cam_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
    print(f"Webcam native resolution : {cam_width}x{cam_height}")
    print(f"Display resolution       : {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

    # Set up our "Presentation Mode" with multiple dynamic slides
    slides = [Canvas(DISPLAY_HEIGHT, DISPLAY_WIDTH)]
    backgrounds = [np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8) * 255]
    original_images = [None]
    image_scales = [1.0]
    current_idx = 0
    detector = HandDetector()

    # Create a resizable named window and size it upfront
    cv.namedWindow("Airdraw", cv.WINDOW_NORMAL)
    cv.resizeWindow("Airdraw", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    print("Controls:")
    print(" 'n'=Next Slide | 'p'=Prev Slide | 'x'=Delete Slide | 'i'=Insert Image | '[' / ']'=Scale")
    print(" 'c'=Color | 'h'=Shape | 's'=Save | '=' / '-'=Thickness | 'a'=Clear | 'q'=quit")
    print(" ',' / '.'=Eraser Size | 'z'=Undo | 'y'=Redo | 'd'=Dark Mode")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        # The webcam is ONLY for hand tracking now. We do not display the AR feed.
        frame = cv.flip(frame, 1)  # 1 = flip horizontally (mirror)
        frame = cv.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv.INTER_LINEAR)
        gesture_metadata = detector.get_gesture_metadata(frame)

        # Get the current slide and its background image
        canvas = slides[current_idx]
        if original_images[current_idx] is None:
            # Generate the background color dynamically based on Current Dark/Light mode
            if getattr(canvas, 'dark_mode', False):
                display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            else:
                display_frame = np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8) * 255
            backgrounds[current_idx] = display_frame
        else:
            display_frame = backgrounds[current_idx].copy()
        
        # Draw on our whiteboard, using the hand tracking from the invisible webcam frame
        rendered_frame = canvas.update_and_draw(display_frame, gesture_metadata)
        
        # Add slide counter watermark HUD
        overlay = rendered_frame.copy()
        
        # Determine the HUD panel size programmatically
        hud_text = f"Slide {current_idx + 1}/{len(slides)}  |  Color: {canvas.color.name}  |  Shape: {canvas.shape.name}  |  Pen: {canvas.thickness}px  |  Eraser: {canvas.eraser_thickness}px"
        (tw, th), _ = cv.getTextSize(hud_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        hud_margin = 15
        hud_x1 = (DISPLAY_WIDTH - tw) // 2 - hud_margin
        hud_y1 = DISPLAY_HEIGHT - th - (hud_margin * 2) - 15
        hud_x2 = hud_x1 + tw + (hud_margin * 2)
        hud_y2 = DISPLAY_HEIGHT - 10
        
        from canvas import draw_rounded_rect
        draw_rounded_rect(overlay, (hud_x1, hud_y1+5), (hud_x2, hud_y2+5), (0, 0, 0), -1, 15) # Shadow
        draw_rounded_rect(overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), (40, 40, 40) if getattr(canvas, 'dark_mode', False) else (245, 245, 245), -1, 15)
        rendered_frame = cv.addWeighted(overlay, 0.85, rendered_frame, 0.15, 0)
        
        text_color = (255, 255, 255) if getattr(canvas, 'dark_mode', False) else (30, 30, 30)
        cv.putText(rendered_frame, hud_text, (hud_x1 + hud_margin, hud_y2 - hud_margin + 2),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv.LINE_AA)

        cv.imshow("Airdraw", rendered_frame)

        stroke = cv.waitKey(1) & 0xFF
        if stroke == ord('a'):              # clear all drawings on current slide
            canvas.clear_all()
            
        if stroke == ord('.'):              # increase eraser thickness
            canvas.eraser_thickness = min(150, getattr(canvas, 'eraser_thickness', 40) + 10)
            
        if stroke == ord(','):              # decrease eraser thickness
            canvas.eraser_thickness = max(10, getattr(canvas, 'eraser_thickness', 40) - 10)
            
        if stroke == ord('z') or stroke == 26:  # undo
            canvas.undo()
            
        if stroke == ord('y') or stroke == 25:  # redo
            canvas.redo()
            
        if stroke == ord('d'):              # toggle dark mode
            canvas.toggle_dark_mode()
            
        if stroke == ord('c'):              # cycle color
            try:
                curr_c_idx = canvas.colors.index(canvas.color)
            except ValueError:
                curr_c_idx = 0
            canvas.color = canvas.colors[(curr_c_idx + 1) % len(canvas.colors)]
            
        if stroke == ord('h'):              # cycle shape
            try:
                curr_s_idx = canvas.shapes.index(canvas.shape)
            except ValueError:
                curr_s_idx = 0
            canvas.shape = canvas.shapes[(curr_s_idx + 1) % len(canvas.shapes)]
        
        if stroke == ord('n'):              # next slide
            current_idx += 1
            if current_idx >= len(slides):
                slides.append(Canvas(DISPLAY_HEIGHT, DISPLAY_WIDTH))
                backgrounds.append(np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8) * 255)
                original_images.append(None)
                image_scales.append(1.0)
        
        if stroke == ord('p'):              # prev slide
            current_idx = max(0, current_idx - 1)
            
        if stroke == ord('x'):              # delete slide
            if len(slides) > 1:
                slides.pop(current_idx)
                backgrounds.pop(current_idx)
                original_images.pop(current_idx)
                image_scales.pop(current_idx)
                if current_idx >= len(slides):
                    current_idx = len(slides) - 1
            else:
                slides[0].clear_all()
                backgrounds[0] = np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8) * 255
                original_images[0] = None
                image_scales[0] = 1.0

        if stroke == ord('i'):              # insert image (using a simple dialog)
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            filepath = filedialog.askopenfilename(title="Select Slide Background Image",
                                                  filetypes=[("Images", "*.png *.jpg *.jpeg")])
            root.destroy()
            if filepath:
                img = cv.imread(filepath)
                if img is not None:
                    original_images[current_idx] = img
                    image_scales[current_idx] = 1.0
                    backgrounds[current_idx] = get_scaled_background(img, 1.0, DISPLAY_WIDTH, DISPLAY_HEIGHT)

        if stroke == ord(']'):              # image scale up
            if original_images[current_idx] is not None:
                image_scales[current_idx] += 0.1
                backgrounds[current_idx] = get_scaled_background(original_images[current_idx], image_scales[current_idx], DISPLAY_WIDTH, DISPLAY_HEIGHT)
        
        if stroke == ord('['):              # image scale down
            if original_images[current_idx] is not None:
                image_scales[current_idx] = max(0.1, image_scales[current_idx] - 0.1)
                backgrounds[current_idx] = get_scaled_background(original_images[current_idx], image_scales[current_idx], DISPLAY_WIDTH, DISPLAY_HEIGHT)

        if stroke == 19 or stroke == ord('s'):  # save all slides (Ctrl+S or s)
            print("Exporting presentation...")
            for i in range(len(slides)):
                slide_export = slides[i].render_clean(backgrounds[i])
                fname = f"presentation_slide_{i+1}.png"
                cv.imwrite(fname, slide_export)
                print(f"-> Saved {fname}")
            print("Export complete!")

        if stroke == ord('='):              # increase thickness
            canvas.thickness = min(20, canvas.thickness + 1)
        if stroke == ord('-'):              # decrease thickness
            canvas.thickness = max(1, canvas.thickness - 1)
        if stroke == ord('q') or stroke == 27:   # quit
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
