import cv2
import numpy as np

def process_state_image(state, step):
    state = np.array(state)
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    # state = cv2.resize(state, (30, 30))
    # cv2.imwrite(f"images/state_{step}.png", state)
    state = state.astype(float)
    state /= 255.0
    return state

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))
