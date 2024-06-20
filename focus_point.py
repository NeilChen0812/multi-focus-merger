import numpy as np
from sharpness_comparison import get_sharpness_values


def find_focus_points(images, sharpness_threshold=2000, diff_threshold=100):
    sharpness_values = get_sharpness_values(images)
    sharpness_values = np.array(sharpness_values)
    diff = np.insert(np.diff(sharpness_values), 0, 0)
    focus_points = np.where(
        (np.abs(diff) < diff_threshold) & (sharpness_values > sharpness_threshold))[0]
    return focus_points


if __name__ == '__main__':
    # parameters
    video_name = 'P1500718'
    video_path = f'videos/{video_name}.MP4'
    sampling_interval = 1

    # images = film2image_list(video_path, sampling_interval)
    # sharpness_values = get_sharpness_values(images)
    # print(sharpness_values)
    sharpness_values = [245, 244, 230, 228, 236, 231, 235, 235, 243, 245, 244, 256, 298, 297, 282, 274, 274, 290, 283, 284, 297, 285, 287, 316, 317, 317, 306, 297, 306, 337, 330, 342, 356, 350, 350, 364, 367, 369, 369, 366, 366, 392, 383, 399, 449, 445, 447, 525, 553, 563, 577, 577, 676, 792, 814, 817, 911, 1040, 1267, 1338, 1794, 1839, 2221, 3120, 3272, 3631, 3884, 3908, 3910, 3950, 3992, 4056, 4032, 4008, 3995, 3976, 3960, 3958, 3914, 3882, 3892, 3810, 3835, 3909, 3807, 3781, 3732, 3694, 3689, 3754, 3830, 3842, 3894, 3946,
                        4346, 4309, 4376, 4485, 4450, 4582, 4736, 5071, 5211, 5166, 5273, 5146, 4699, 4669, 4129, 4079, 3688, 3690, 3661, 3600, 3567, 3562, 3549, 3493, 3407, 3411, 3158, 2757, 2401, 1950, 1860, 1566, 1203, 1025, 1032, 823, 749, 695, 635, 594, 568, 508, 487, 480, 428, 415, 428, 391, 384, 401, 368, 361, 336, 318, 314, 328, 317, 314, 328, 316, 315, 327, 317, 309, 288, 273, 273, 290, 274, 273, 291, 280, 279, 296, 285, 283, 269, 263, 263, 278, 270, 269, 283, 275, 275, 289, 281, 282, 268, 264, 264, 276, 269, 270, 283, 269, 268, 290]

    focus_points = find_focus_points(sharpness_values)
    print(focus_points)
