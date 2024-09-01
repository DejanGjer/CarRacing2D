import cv2
import numpy as np

def process_state_image(frame, image_size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # if image size is not 96x96, resize it
    if image_size != (96, 96):
        frame = cv2.resize(frame, image_size)
    # cv2.imwrite(f"images/state_{step}.png", state)
    frame = frame.astype(float)
    frame /= 255.0
    return frame

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    return frame_stack
    # # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    # return np.transpose(frame_stack, (1, 2, 0))
