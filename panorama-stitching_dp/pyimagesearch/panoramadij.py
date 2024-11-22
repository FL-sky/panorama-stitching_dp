# import the necessary packages
import numpy as np
import imutils
import cv2
from queue import PriorityQueue

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
	color_difference_energy = np.sum((I1.astype(np.float32) - I2.astype(np.float32)) ** 2, axis=2)

	# 组合能量
	total_energy = 0.5 * (gradient_energy1 + gradient_energy2) + 0.5 * color_difference_energy
	total_energy = color_difference_energy
	return total_energy


iN = 1288
iM = 952
N = iN * iM
M = (N + iM) * 4

# 初始化
n, m = 0, 0
A = np.zeros((iN, iM), dtype=int)  # A 图
B = np.zeros((iN, iM), dtype=int)  # B 图
C = np.zeros((iN, iM), dtype=int)  # C 图
ecnt = 0
t = [0] * M
nxt = [0] * M
head = [0] * M
val = [0] * M

dis = [float('inf')] * N
pre = [-1] * N

S, T = 0, 0
inque = [0] * (N + 1)


def init(x):
	global ecnt
	ecnt = 0
	global head
	head = [0] * x


def addedge(from_node, to_node, distance):
	global ecnt
	ecnt+=1
	t[ecnt] = to_node
	nxt[ecnt] = head[from_node]
	head[from_node] = ecnt
	val[ecnt] = distance
	ecnt += 1

	t[ecnt] = from_node
	nxt[ecnt] = head[to_node]
	head[to_node] = ecnt
	val[ecnt] = distance

def addedge2(from_node, to_node):
	global ecnt
	ecnt+=1
	t[ecnt] = to_node
	nxt[ecnt] = head[from_node]
	head[from_node] = ecnt

	ecnt += 1

	t[ecnt] = from_node
	nxt[ecnt] = head[to_node]
	head[to_node] = ecnt


def dijkstra(start, end):
	global dis, pre,T
	pre[:] = [-1] * N
	dis[:] = [float('inf')] * N
	dis[start] = 0
	pq = PriorityQueue()
	pq.put((0,start))
	vis = [False]*(end+1)

	while pq.qsize()>0:
		u = pq.get()[1]
		if vis[u]:
			continue
		vis[u]=True
		if u == T:
			break
		i = head[u]
		while i>0:
			v = t[i]
			w = val[i]
			if not vis[v] and dis[v] > dis[u] + w:
				dis[v] = dis[u] + w
				pre[v] = u
				pq.put((dis[v], v))
			i=nxt[i]


def bfsgetC(start):
	global C
	C[:] = B.copy()
	inque[:] = [False] * (n * m)

	que = [start]
	inque[start] = True
	x, y = divmod(start, m)
	C[x][y] = A[x][y]
	qfront, qtail = 0, 1
	directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

	while qfront < qtail:
		ele = que[qfront]
		qfront += 1
		x, y = divmod(ele, m)

		for dx, dy in directions:
			nx, ny = x + dx, y + dy
			if 0 <= nx < n and 0 <= ny < m:
				nele = nx * m + ny
				if inque[nele]:
					continue
				can = True
				j = head[ele]
				while j>0:
					v = t[j]
					if v == nele:
						can = False
						break
					j=nxt[j]
				if can:
					que.append(nele)
					qtail += 1
					inque[nele] = True
					C[nx][ny] = A[nx][ny]


def printl():
	global S,pre
	u = T
	ret = []
	while u != S:
		p = pre[u]
		realp = p
		if u == T:
			j = p % (m - 1)
			i = n - 1
			ret.append((i * m + j - 1, i * m + j))
		elif p == S:
			ret.append((u - 1, u))
		else:
			if abs(p - u) == 1:  # 竖线
				if p > u:
					p, u = u, p
				i, j = divmod(p, m - 1)
				ret.append((i * m + j, i * m + j + m))
			elif abs(u - p) == m - 1:  # 横线
				if u < p:
					p = u
				i, j = divmod(p, m - 1)
				ret.append(((i + 1) * m + j - 1, (i + 1) * m + j))

		addedge2(ret[-1][0], ret[-1][1])  # 记录割线
		u=realp

	bfsgetC(0)

	# for row in C:
	# 	print(" ".join(f"{int(val):03d}" for val in row))

def find_seam(A,B):
	A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
	B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)

	n, m = A.shape
	start = 0
	end = (n - 1) * (m - 1) + 1
	S = start
	T = end
	for j in range(1, m):
		w = abs(A[0][j - 1] - B[0][j - 1]) + abs(A[0][j] - B[0][j])
		addedge(start, j, w)

	for i in range(1, n - 1):
		for j in range(1, m):
			w = abs(A[i][j - 1] - B[i][j - 1]) + abs(A[i][j] - B[i][j])
			u = (i - 1) * (m - 1) + j
			v = u + (m - 1)
			addedge(u, v, w)

	for j in range(1, m):
		w = abs(A[n - 1][j - 1] - B[n - 1][j - 1]) + abs(A[n - 1][j] - B[n - 1][j])
		u = (n - 2) * (m - 1) + j
		addedge(u, end, w)

	for i in range(n - 1):
		for j in range(1, m - 1):
			w = abs(A[i][j] - B[i][j]) + abs(A[i + 1][j] - B[i + 1][j])
			u = i * (m - 1) + j
			v = u + 1
			addedge(u, v, w)


	dijkstra(start, end)

	with open("cutdij0617.txt", "w") as f:
		f.write(f"{dis[end]}\n")
		printl()


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
		for i in range(imageB.shape[1],0,-1):
			ival = result[0,i,0]+result[0,i,1]+result[0,i,2]
			if ival <= 0:
				leftborder = i
				break

		imageLeft = imageB[:, leftborder + 1:imageB.shape[1]]
		imageRight = result[:, leftborder + 1:imageB.shape[1]]

		# energy = compute_energy(imageLeft, imageRight)

		# 找到拼接缝
		seam = find_seam(imageLeft,imageRight)



		for i in range(0,imageB.shape[0]):
			result[i,:leftborder+seam[i]] = imageB[i,:leftborder+seam[i]]

		# check to see if the keypoint matches should be visualized
		print("showMatches=",showMatches)
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