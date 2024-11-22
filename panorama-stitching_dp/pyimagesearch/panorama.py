# import the necessary packages
import numpy as np
import imutils
import cv2


def compute_energy(I1, I2):
    # 转换为灰度图
    gray1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # 计算梯度
    sobel_x1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)

    # # 计算梯度能量
    gradient_energy1 = np.abs(sobel_x1) + np.abs(sobel_y1)
    gradient_energy2 = np.abs(sobel_x2) + np.abs(sobel_y2)
    # 计算梯度能量
    # gradient_energy1 = sobel_x1**2 + sobel_y1**2
    # gradient_energy2 = sobel_x2**2 + sobel_y2**2

    # 计算颜色差异能量
    color_difference_energy = np.sum(abs(I1.astype(np.float64) - I2.astype(np.float64)) ** 2.5, axis=2)

    # 组合能量
    # total_energy = 0.5 * (gradient_energy1 + gradient_energy2) + 0.5 * color_difference_energy
    total_energy = color_difference_energy
    # total_energy = gradient_energy1 + gradient_energy2
    return total_energy.astype(np.float64)

def compute_energy1(I1, I2):
    # 转换为灰度图
    hsv1 = cv2.cvtColor(I1, cv2.COLOR_RGB2YUV)
    hsv2 = cv2.cvtColor(I2, cv2.COLOR_RGB2YUV)

    # 计算颜色差异能量
    # color_difference_energy = np.sum(abs(hsv1[:,:,1:].astype(np.float64) - hsv2[:,:,1:].astype(np.float64)) ** 2, axis=2)
    color_difference_energy = np.sum(abs(hsv1[:, :, :].astype(np.float64) - hsv2[:, :, :].astype(np.float64)) ** 2, axis=2)

    total_energy = color_difference_energy
    return total_energy.astype(np.float64)

def find_seam(E):
    rows, cols = E.shape
    dp = np.ones_like(E) * 1e9
    dp[0, :] = E[0, :]

    for y in range(1, rows):
        for x in range(cols):
            min_energy = dp[y - 1, x]  # 从上方
            if x > 0:
                min_energy = min(min_energy, dp[y - 1, x - 1], dp[y - 1, x])
            if x < cols - 1:
                min_energy = min(min_energy, dp[y - 1, x + 1])
            dp[y, x] = E[y, x] + min_energy
        for x in range(cols - 2, -1, -1):
            dp[y, x] = min(dp[y, x], dp[y, x + 1] + E[y, x])

    # 回溯找到最小能量路径
    seam = []
    min_index = np.argmin(dp[-1, :])
    th, tw = rows - 1, min_index
    # for y in range(rows - 1, -1, -1):
    rt = [1e9]*rows
    while th>=0:
        seam.append((th, tw))
        rt[th] = min(rt[th], tw)

        mi = dp[th-1,tw]
        if th > 0:
            if tw > 0:
                mi = min(mi, dp[th - 1, tw - 1], dp[th, tw - 1])
            if tw < cols - 1:
                mi = min(mi, dp[th - 1, tw + 1], dp[th, tw + 1])

            if mi == dp[th-1,tw]:
                th -= 1
            elif mi == dp[th - 1, tw - 1]:
                th-=1
                tw-=1
            elif mi == dp[th - 1, tw + 1]:
                th -= 1
                tw += 1
            elif mi == dp[th, tw - 1]:
                while tw > 0 and mi == dp[th, tw - 1]:
                    tw -= 1
                    seam.append((th, tw))
                    rt[th] = min(rt[th], tw)

            elif mi == dp[th, tw + 1]:
                while tw < cols - 1 and mi == dp[th, tw + 1]:
                    tw+=1
                    seam.append((th, tw))
                    rt[th] = min(rt[th], tw)
        else:
            break

    seam.reverse()
    # return seam
    return rt


def draw_seam(I1, I2, seam):
    for y, x in enumerate(seam):
        I1[y, x] = [0, 255, 0]  # 将接缝位置标记为绿色
    return I1


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        leftborder = 0
        for i in range(imageB.shape[1], 0, -1):
            ival = result[0, i, 0] + result[0, i, 1] + result[0, i, 2]
            if ival <= 0:
                leftborder = i
                break

        imageLeft = imageB[:, leftborder + 1:imageB.shape[1]]
        imageRight = result[:, leftborder + 1:imageB.shape[1]]

        energy = compute_energy(imageLeft, imageRight)

        # 找到拼接缝
        seam = find_seam(energy)

        # 绘制接缝
        # stitched_image = draw_seam(imageLeft.copy(), imageRight, seam)
        #
        # # 显示结果
        # cv2.imshow('Stitched Image', stitched_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        ltr = 3
        for i in range(0, imageB.shape[0]):
            for j in range(-ltr,ltr):
                xv = ((1.0 * result[i, leftborder + seam[i] + j] + imageB[i, leftborder + seam[i] + j]) / 2)
                result[i, leftborder + seam[i]+j] = xv
            result[i, :leftborder + seam[i]-ltr] = imageB[i, :leftborder + seam[i]-ltr]

            # result[i,leftborder] = [0,255,0]
            # result[i, imageB.shape[1]] = [0,255,0]
            #
            # result[i, leftborder + seam[i]] = [0, 0, 255]
            for j in range(leftborder + seam[i]-1, imageB.shape[1]):
                if np.sum(result[i, j]) == 0:
                    # result[i, j] = 0.1 * imageB[i, j] + 0.3 * result[i - 1, j - 1] + 0.3 * result[i - 1, j] + 0.3 * \
                    #                result[i - 1, j + 1]
                    k = j
                    while k<imageB.shape[1]:
                        result[i, k] = 0.1 * imageB[i, k] + 0.3 * result[i - 1, k - 1] + 0.3 * result[i - 1, k] + 0.3 *  result[i - 1, k + 1]
                        k+=1
                    break


            # result[i, leftborder + seam[i]] = [0, 0, 255]

        # feng=1
        # result[0:imageB.shape[0], 0:imageB.shape[1]-feng] = imageB[:,:-feng]
        # result[0:imageB.shape[0], imageB.shape[1] - feng:imageB.shape[1]] = imageB[:,-feng:]*0.5+result[0:imageB.shape[0], imageB.shape[1] - feng:imageB.shape[1]]*0.5

        # check to see if the keypoint matches should be visualized
        print("showMatches=", showMatches)
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            ### https://www.bilibili.com/video/BV13v411E7M7/?spm_id_from=333.337.search-card.all.click&vd_source=32226861dfac47dcce527b8d223d9c5c   SIFT
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis