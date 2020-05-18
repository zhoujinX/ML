from skimage import io
from numpy import *

def KMeansInitCentroids(X,K):
    randidx = randperm(size(X,1))
    centroids = X[randidx[1:K],:]
    return centroids

def findClosestCentroids(X,centroids):
    for i in size(X,1):
        distance = 999999999
        for j in K:
            dist = sum((X[i,:]-centroids[j,:])^2)
            if dist < distance:
                idx[i] = j
            end
        endfor
    endfor
    return idx   

def runMeans(X,initial_centroids,max_iters):
    count = zeros(K,1)
    for i in max_iters:
        id = idx[i]
        centroids[id,:] = centroids[id,:] + X[i,:]
        count[id] = count(id) + 1
    endfor
    initial_centroids = centroids/count

#图片数据处理
A = double(io.imread('36042919970222033X.jpg'))  #调用imread读取图片
A = A/255   #将像素值处理到0-1
img_size = size(A)  #将数据从3维降到2维
X = reshape(A, img_size(1)*img_size(2),3)

#获取聚类中心
#随机初始化
K = 16
initial_centroids = KMeansInitCentroids(X,K)
#调用k均值算法
max_iters = 10
[centroids,idx] = runMeans(X,initial_centroids,max_iters)

#压缩,调用分配聚类中心
idx = findClosestCentroids(X,centroids)

#解压缩
#通过聚类表获取原始像素值
X_recovered = centroids[idx,:]
#将数据升至3维
X_recovered = reshape(X_recovered,img_size(1),img_size(2),3)

