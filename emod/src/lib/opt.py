import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def visualize_optical_flow_lucas_kanade(frame, flow_vectors_with_p0):
    plt.figure(figsize=(10, 10))
    plt.imshow(frame, cmap='gray')
    p0, flow_vector = flow_vectors_with_p0
    for i, (dx, dy) in enumerate(flow_vector):
        x, y = p0[i].ravel()
        plt.arrow(x, y, dx, dy, color='red', head_width=5, head_length=7)
    plt.axis('off')
    plt.show()

def filter_flow_by_magnitude(flow, threshold=1.0):
    """
    Filters optical flow vectors by magnitude.

    Parameters:
    - flow: The optical flow vectors of shape [num_vectors, 2], where each vector is [dx, dy].
    - threshold: Magnitude threshold for filtering.

    Returns:
    - A tensor of the same shape as flow, with vectors having magnitude below the threshold set to [0, 0].
    """
    # Calculate magnitude of each vector
    magnitude = torch.sqrt(flow[1][:, 0]**2 + flow[1][:, 1]**2)  # Compute magnitude along the correct dimension

    # Create mask where magnitude is greater than threshold
    mask = magnitude > threshold

    # Filter flow vectors: set vectors with magnitude below the threshold to zero
    flow_filtered = flow[1].clone()
    flow_filtered[~mask, :] = 0  # Apply mask correctly
    
    return (flow[0], flow_filtered)

def load_frames(paths, loader=torch.load, progress_bar=True):
    if progress_bar:
        return [loader(path) for path in tqdm(paths)]
    else:
        return [loader(path) for path in paths]

def convert_us_to_ms(us):
    return us // 1000

def find_index_range(time_array, start, end):
    idx_start = np.searchsorted(time_array, start, side='left')
    idx_end = np.searchsorted(time_array, end, side='right')
    return idx_start, idx_end

def process_events(t_start_us, t_end_us, event_data):
    t_start_ms = convert_us_to_ms(t_start_us)
    t_end_ms = convert_us_to_ms(t_end_us)
    idx_start, idx_end = find_index_range(event_data['t'], t_start_ms, t_end_ms)
    if idx_start == idx_end:  # No events in the range
        return None
    return {key: val[idx_start:idx_end] for key, val in event_data.items()}

def torch_divmod(x, y):
    return torch.div(x, y, rounding_mode='trunc'), torch.remainder(x, y)

def separate_event_data(event_data, width):
    indices = torch.tensor(event_data['index'][-1])
    y_coords, x_coords = torch_divmod(indices, width)
    timestamps = torch.tensor(event_data['last_timestamps'])
    polarity = torch.tensor(event_data['stacked_polarity'][-1])
    sorted_indices = timestamps.argsort()
    midpoint = len(timestamps) // 2
    return (x_coords[sorted_indices[:midpoint]], y_coords[sorted_indices[:midpoint]], polarity[:midpoint]), \
           (x_coords[sorted_indices[midpoint:]], y_coords[sorted_indices[midpoint:]], polarity[midpoint:])

def create_frame(events, width, height):
    frame = torch.zeros((height, width), dtype=torch.uint8)
    x, y = events
    frame[y, x] += 1  # Assuming each event increases the intensity by 1
    return frame

def create_frame_from_events(x_coords, y_coords, polarity, width, height):
    """
    Create a grayscale image from event data.
    """
    frame = torch.zeros((height, width), dtype=torch.int16, device=x_coords.device)
    # print()
    # print(x_coords.dtype, y_coords.dtype, polarity.dtype)
    # print()
    # Accumulate polarity in frame creation, with positive and negative polarity handled
    for x, y, p in zip(x_coords, y_coords, polarity):
        frame[y, x] += p
    return frame.to(torch.uint8).cpu().numpy()

def estimate_optical_flow(event_halves, width, height):
    """
    Estimates optical flow between two event-based frames constructed from halves of event data.

    Args:
    event_halves ((tuple, tuple)): Each tuple contains x_coords, y_coords, and polarity information for half of the events.
    width (int): Width of the frames to be created.
    height (int): Height of the frames to be created.
    feature_params (dict): Parameters for cv2.goodFeaturesToTrack.
    lk_params (dict): Parameters for cv2.calcOpticalFlowPyrLK.
    
    Returns:
    tuple: Contains old points and flow vectors.
    """
    feature_params = {
        'maxCorners': 256,
        'qualityLevel': 0.04,  # Looking for higher quality features
        'minDistance': 10,
        'blockSize': 9  # Larger block size for more robust averaging
    }

    lk_params = {
        'winSize': (15, 15),  # Very large window to smooth over noise
        'maxLevel': 1,        # Fewer levels to avoid noise amplification
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.05)  # More lenient in finding convergence
    }

    # Convert event data to frames
    old_half, new_half = event_halves
    old_frame = create_frame_from_events(*old_half, width, height)
    new_frame = create_frame_from_events(*new_half, width, height)
    
    # Apply Gaussian blur to the frames
    # old_frame = cv2.GaussianBlur(old_frame, (3, 3), 0)
    # new_frame = cv2.GaussianBlur(new_frame, (3, 3), 0)
    
    # Detect good features in the old frame
    p0 = cv2.goodFeaturesToTrack(old_frame, mask=None, **feature_params)
    
    # Ensure p0 is not None to proceed with optical flow calculation
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, p0, None, **lk_params)
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            flow_vectors = good_new - good_old
            return ((old_frame, new_frame), (torch.from_numpy(good_old).to(old_half[0].device), torch.from_numpy(flow_vectors).to(old_half[0].device)))
    
    # Return empty if no features or optical flow can be calculated
    return (torch.from_numpy(np.array([])), torch.from_numpy(np.array([])))

def create_full_flow_map(p0, flow_vectors, image_shape):
    if len(p0.shape) == 2:
        p0 = p0.unsqueeze(0)
        flow_vectors = flow_vectors.unsqueeze(0)
    b, n, _ = p0.shape
    # print("p0 size:", p0.shape,"flow_vectors size: ", flow_vectors.shape)
    h, w = image_shape
    full_flow_map = torch.zeros((b, 2, h, w), dtype=flow_vectors.dtype, device=flow_vectors.device)
    
    indices_y = p0[:, :, 1].long()
    indices_x = p0[:, :, 0].long()
    
    # Flatten the indices to use with index_put_ or similar functions
    linear_indices = indices_y * w + indices_x  # Convert 2D indices to 1D
    # print("linear_indices size: ", linear_indices.shape)
    linear_indices = linear_indices.unsqueeze(1)  # Shape becomes [b, 1, n]

    # We need to adjust linear indices to match the dimension of full_flow_map when viewed as [b, 2, h*w]
    full_flow_map = full_flow_map.view(b, 2, h * w)  # View as 2D for easier indexing
    # Expand linear indices along the flow component axis to match flow_vectors' second dimension
    linear_indices = linear_indices.expand(-1, 2, -1)  # Now shape is [b, 2, n]

    # Scatter flow vectors into the full flow map; flow_vectors need to be permuted correctly
    flow_vectors = flow_vectors.permute(0, 2, 1)  # Adjust flow_vectors shape to [b, n, 2]
    flow_vectors = flow_vectors.contiguous().view(b, n, 2)  # Ensure it's contiguous and correctly shaped

    for i in range(2):  # We need to handle each component separately due to different indexing
        full_flow_map[:, i, :].scatter_(1, linear_indices[:, i, :], flow_vectors[:, :, i])

    full_flow_map = full_flow_map.view(b, 2, h, w)  # Reshape back to original dimensions
    
    return full_flow_map


def warp_features(features, flow_map):
    """
    Warp features using a full optical flow map.

    Args:
    features (Tensor): Tensor of shape [b, h, w], image-like features.
    flow_map (Tensor): Full flow map of shape [b, 2, h, w].

    Returns:
    Tensor: Warped features of the same shape as input features.
    """
    b, c, h, w = features.size()

    # Create grid for each pixel
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
    grid = torch.stack((grid_x, grid_y), dim=-1).to(flow_map.device)  # [h, w, 2]
    # print(grid.shape)
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [b, h, w, 2]
    

    # Adjust for meshgrid indexing differences if necessary
    # grid = grid.permute(0, 2, 1, 3)  # Swap y and x to match 'xy' Cartesian indexing if needed

    # Normalize flow map to [-1, 1]
    normalized_flow_map = torch.clone(grid).to(flow_map.device)
    # print(grid.shape)
    normalized_flow_map[:, :, :, 0] += 2 * flow_map[:, 0] / (w - 1)
    normalized_flow_map[:, :, :, 1] += 2 * flow_map[:, 1] / (h - 1)
    
    # print(features.device, normalized_flow_map.device)

    # Warp features using the normalized grid
    warped_features = F.grid_sample(features, normalized_flow_map, mode='bilinear', padding_mode='zeros', align_corners=False)
    return warped_features