import cv2
import numpy as np
import torch
cap = cv2.VideoCapture('/Users/vishal/Documents/POSE/ONE/pexelscom_pavel_danilyuk_basketball_hd.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC, 100)
ret, frame1 = cap.read()
if ret:
    cv2.imwrite('frame1.jpg', frame1)
else:
    print("Error: failed to capture frames")

cap.set(cv2.CAP_PROP_POS_MSEC, 133.36)
ret, frame2 = cap.read()
cap.release()
if ret:
    cv2.imwrite('frame2.jpg', frame2)
else:
    print("Error: failed to capture frames")

def compute_motion(flow):
    disp = flow.copy()
    disp[..., 0] += np.arange(disp.shape[1])
    disp[..., 1] += np.arange(disp.shape[0])[:, np.newaxis]
    src_pts = disp.reshape(-1, 2).astype(np.float32)
    dst_pts = np.array([[x, y] for y in range(disp.shape[0])
                       for x in range(disp.shape[1])], np.float32)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    return M


def compute_flow(prev_frame, next_frame):
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    ang_norm = ang * 180 / np.pi / 2
    flow[..., 0], flow[..., 1] = cv2.polarToCart(mag_norm, ang_norm)

    return flow


def estimate_image_at_times(img_tk, img_t0):
    flow_t0_to_tk = compute_flow(img_t0, img_tk)

    M_tk_to_t0 = compute_motion(flow_t0_to_tk)

    h, w = img_t0.shape[:2]
    img_tk_to_t0 = cv2.warpPerspective(img_tk, M_tk_to_t0, (w, h))

    flow_t0_to_tk_to_t0 = compute_flow(img_t0, img_tk_to_t0)
    M_t0_to_tk_to_t0 = compute_motion(flow_t0_to_tk_to_t0)

    img_tk_to_t0_refined = cv2.warpPerspective(
        img_tk, M_t0_to_tk_to_t0.dot(M_tk_to_t0), (w, h))

    alpha = 0.5
    img_t_1ms_est = cv2.addWeighted(
        img_tk_to_t0, alpha, img_tk_to_t0_refined, 1 - alpha, 0)
    img_t_1s_est = cv2.addWeighted(img_tk_to_t0, 0.5, img_t0, 0.5, 0)

    return img_t_1ms_est, img_t_1s_est


if __name__ == '__main__':
    img1 = cv2.imread('ONE/frame1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('ONE/frame2.jpg', cv2.IMREAD_GRAYSCALE)

    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.resize(img2, (512, 512))

    cv2.imwrite('ONE/frame1.jpg', img1)
    cv2.imwrite('ONE/frame2.jpg', img2)

    img1 = 'ONE/frame1.jpg'
    img2 = 'ONE/frame2.jpg'

    img_t_33 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    img_t = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    img_t_1ms_est, img_t_1s_est = estimate_image_at_times(img_t_33, img_t)

    cv2.imwrite('image_t_1ms_est.png', img_t_1ms_est)
    cv2.imwrite('image_t_1s_est.png', img_t_1s_est)
