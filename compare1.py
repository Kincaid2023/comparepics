
import cv2
import numpy as np
import os
from skimage.metrics import mean_squared_error



"""
1. divide_image method: divide the two images into NxN blocks
2. compare:  compare the blurness of each blocks on the same place,and find out the most different 3 blocks in blurenss.
3. merge the picture: first pic on left, second on right, all with their name on left top.
4. mark the top 3 block on the merged pic: if  block1 is more blur on pic 1, mark it on the pic1 of the merged picture, if block2 is more blure on pic 2, mark it on the pic2 of the merged picture.

"""
def divide_image(img, N):
    """Divides the given image into NxN blocks"""
    h, w = img.shape[:2]
    block_h, block_w = h // N, w // N
    blocks = []
    for i in range(N):
        for j in range(N):
            block = img[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            blocks.append(block)
    return blocks

def compare_blocks(blocks1, blocks2):
    """Compares the blurness of the corresponding blocks in two sets"""
    scores = []
    for i in range(len(blocks1)):
        score = mean_squared_error(cv2.Laplacian(blocks1[i], cv2.CV_64F),
                                   cv2.Laplacian(blocks2[i], cv2.CV_64F))
        scores.append(score)
    return scores

def mark_differences(img1, img2, N, top_n):
    """Marks the top n differences in blur in the merged image"""
    blocks1 = divide_image(img1, N)
    blocks2 = divide_image(img2, N)
    scores = compare_blocks(blocks1, blocks2)
    top_n_blocks = np.argsort(scores)[-top_n:][::-1]  # indices of top n blocks with highest difference in blur
    merged = np.concatenate((img1, img2), axis=1)
    filename1 = os.path.splitext(os.path.basename("pic-4m.png"))[0]
    filename2 = os.path.splitext(os.path.basename("pic-6m.png"))[0]
    merged = cv2.putText(merged, filename1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    merged = cv2.putText(merged, filename2, (img1.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    h, w = img1.shape[:2]
    block_h, block_w = h // N, w // N
    for i in top_n_blocks:
        x = (i % N) * block_w
        y = (i // N) * block_h
        if scores[i] > np.median(scores):
            if i < len(blocks1):
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            merged = cv2.rectangle(merged, (x, y), (x+block_w, y+block_h), color, 2)
            merged = cv2.putText(merged, f'{i+1}', (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return merged

# Example usage
img1 = cv2.imread('pic-4m.png')
img2 = cv2.imread('pic-6m.png')
N = 10  # number of blocks in each row and column
top_n = 3  # number of top blocks to mark
merged = mark_differences(img1, img2, N, top_n)
# filename = f'merged_{top_n}_{N}_{os.path.splitext(os.path.basename("pic-4m.png"))[0]}_{os.path.splitext(os.path.basename("pic-6m.png"))[0]}.png'
filename = f'merged_{top_n}_{N}_{os.path.splitext(os.path.basename("pic-4m.png"))[0]}_{os.path.splitext(os.path.basename("pic-6m.png"))[0]}.png'
cv2.imwrite(filename, merged)
cv2.imshow('Merged', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
