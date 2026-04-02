import cv2 as cv
import argparse
import sys
import os


def record(fname):
    print("recording ", fname)

    # On Windows, cv.CAP_DSHOW (DirectShow) gives faster and more reliable camera init.
    cam = cv.VideoCapture(0, cv.CAP_DSHOW)

    if not cam.isOpened():
        print("Error: could not open webcam.")
        sys.exit(1)

    # Use whatever width and height possible
    frame_width  = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

    # 'mp4v' requires Apple QuickTime on Windows and may fail silently.
    # 'XVID' with .avi is universally supported on Windows without extra codecs.
    # We auto-select codec/extension based on the user-supplied filename.
    if fname.lower().endswith('.mp4'):
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
    else:
        # Default to .avi / XVID which works on Windows out-of-the-box
        if not fname.lower().endswith('.avi'):
            fname = os.path.splitext(fname)[0] + '.avi'
            print(f"Note: output filename changed to '{fname}' for Windows compatibility.")
        fourcc = cv.VideoWriter_fourcc(*'XVID')

    out = cv.VideoWriter(fname, fourcc, 30.0, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: could not create video writer for '{fname}'.")
        print("Tip: try using a .avi extension instead of .mp4 on Windows.")
        cam.release()
        sys.exit(1)

    print("Recording... Press 'q' to stop.")
    while True:
        ret, img = cam.read()
        if not ret or img is None:
            print("Warning: failed to grab frame, retrying...")
            continue

        img = cv.flip(img, 1)
        out.write(img)
        cv.imshow('Recording', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cam.release()
    cv.destroyAllWindows()
    print("recording complete. shutting down.")


def replay(fname):
    print("replaying", fname)

    cap = cv.VideoCapture(fname)
    print("captured")

    if not cap.isOpened():
        print(f"Error: could not open video file '{fname}'")
        return

    print("Playing back... Press 'q' to stop.")
    while cap.isOpened():
        ret, img = cap.read()

        if not ret or img is None:
            break   # End of file

        cv.imshow('Camera', img)
        print("img size", img.size)

        if cv.waitKey(33) & 0xFF == ord('q'):   # ~30 fps
            break

    cap.release()
    cv.destroyAllWindows()
    print("replay complete", fname)


def main():
    parser = argparse.ArgumentParser(
        prog='data.py',
        description='data collection tools'
    )
    parser.add_argument("-m", "--mode")
    parser.add_argument("-f", "--filename")
    args = parser.parse_args()

    if not args.filename:
        print("Error: --filename is required.")
        sys.exit(1)

    # Accept both .mp4 and .avi; warn if neither
    if not (args.filename.lower().endswith(".mp4") or args.filename.lower().endswith(".avi")):
        print(f"filename({args.filename}) should end with .mp4 or .avi")
        return False

    if args.mode == 'replay':
        replay(args.filename)
    elif args.mode == "record":
        record(args.filename)
    else:
        print(f"data mode must be one of ['replay', 'record'], provided '{args.mode}'")
        return False


if __name__ == "__main__":
    main()
