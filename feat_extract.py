import cv2
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
import numpy as np
from scipy.spatial.transform import Rotation as ROT

fx = 617.4224243164062
fy = 617.7899780273438
cx = 316.72235107421875
cy = 244.21875

K = np.array([[fx ,0, cx],
     [0, fy, cy],
     [0, 0, 1]])

def siftMatching(kp1, des1, kp2, des2):

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Lowe's Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance: # 0.75 is the suggested ratio by Lowe. It can be tuned
            good.append(m)
  
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)
    features_number = src_pts.shape[0]
    return src_pts, dst_pts, features_number

    
    
def recover_angle(src_pts, dst_pts):
  
    model, inliers = ransac(
            (src_pts, dst_pts),
            EssentialMatrixTransform,min_samples=8,
            residual_threshold=1, max_trials=1000
        )
    
    n_inliers = np.sum(inliers)
    print('number of features after ransac =', n_inliers)

    inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
    inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
    if n_inliers > 10:
      inlier_keypoints_left = inlier_keypoints_left[0:10]
      inlier_keypoints_right = inlier_keypoints_right[0:10]
      n_inliers = 10
    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
    #image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
    #cv2.imshow('Matches', image3)
    #cv2.waitKey(0)
    src = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
    dst = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

    E=cv2.findEssentialMat(src, dst, K)
    [R1,R2,t_dec]=cv2.decomposeEssentialMat(E[0])
    estimated_rot1 = ROT.from_matrix(R1)
    [thetax1, thetay1, thetaz1] = estimated_rot1.as_euler('xyz', degrees=True)
    estimated_rot2 = ROT.from_matrix(R2)
    [thetax2, thetay2, thetaz2] = estimated_rot2.as_euler('xyz', degrees=True)
    if abs(thetax1)+abs(thetaz1) > abs(thetax2)+abs(thetaz2): # Choice between R1 and R2, work with the one with small rotation around x and z axis
      if abs(thetax2)<6 and abs(thetaz2)<6 and abs(thetax2)<abs(thetay2) and abs(thetaz2)<abs(thetay2): # The y angle should be greater wrt the other two, which should be nearly 0
        angle = thetay2
      else:
        print('trying again...')
        return recover_angle(src_pts, dst_pts) 
    else:
      if abs(thetax1)<6 and abs(thetaz1)<6 and abs(thetax1)<abs(thetay1) and abs(thetaz1)<abs(thetay1):
        angle = thetay1
      else: 
        print('trying again...')
        return recover_angle(src_pts, dst_pts)
    return angle, src, dst

def main(args=None):
  img1 = cv2.imread('image1.png') 
  img2 = cv2.imread('image2.png')

  sift = cv2.SIFT_create()
  kp1, des1 = sift.detectAndCompute(img1, None)
  kp2, des2 = sift.detectAndCompute(img2, None)
  src_main, dst_main, length = siftMatching(kp1, des1, kp2, des2)
  est_angle = recover_angle(src_main, dst_main)
  print('estimated angle = ',est_angle)


if __name__ == '__main__':
  main()
  