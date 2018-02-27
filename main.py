'''
This script implemets project number 27 Salient object segmentation. for the implementation the method chosen is the methode discribed in article [1]
'''

# imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Global Variables
path_to_image = "MSO/img/COCO_COCO_train2014_000000151534.jpg"
plt.ion()
'''
Classes used for defining the structures and algorithms useful for the implementation
'''

class Bin:
    """
    Class representing a histogram bin
    """
    def __init__(self, pos, is_peak = False):
        self.pos = pos #bin position in the histogram
        self.is_peak = is_peak #True if the bin is a peak

class Hill_climbing:
    """
    Class used for hill climbing algorithm
    """
    def __init__(self, hist, n_bins):
        self.hist = hist #Image histogram
        self.n_bins = n_bins #Number of bins for each dimension of the histogram
        self.not_visited = [] #bins not visited yet (see find_peaks for more details)
        self.peaks_pos = [] #positions of found peaks
        #Fill not_visited with non-zero bins
        for i in range(n_bins):
            for j in range(n_bins):
                for k in range(n_bins):
                    if hist[i,j,k] != 0:
                        self.not_visited.append((i,j,k))
                        
    def cyclic_coord(self, x):
        """
        Returns coordinates of neighbors in one dimension using cyclic
        scheme
        """
        if x == 0:
            min_x = self.n_bins - 1 #we are at the start of the histogram so left neighbor is last bin
            max_x = 1
        elif x == self.n_bins - 1:
            min_x = self.n_bins - 2 #we are at the end of the histogram so right neighbor is first bin
            max_x = 0
        else:
            min_x = x - 1
            max_x = x + 1
        return min_x, max_x 

    def find_neighbors(self, pos):
        """
        Returns neighbor positions and values
        """
        x,y,z = pos
        min_x, max_x = self.cyclic_coord(x)
        min_y, max_y = self.cyclic_coord(y)
        min_z, max_z = self.cyclic_coord(z)
        #Neighbor positions
        neighbors_coord = np.array([[min_x, x, max_x], [min_y, y, max_y], [min_z, z, max_z]])
        #Neighbor histogram values
        neighbors = np.zeros((3,3,3))
        for i, val_x in enumerate(neighbors_coord[0]):
            for j, val_y in enumerate(neighbors_coord[1]):
                for k, val_z in enumerate(neighbors_coord[2]):
                    neighbors[i,j,k] = self.hist[val_x, val_y, val_z]
        return neighbors_coord, neighbors
    
    def find_peaks(self):
        """
        Returns number of peaks in the histogram
        """
        curr_bin = Bin(self.not_visited[0])
        self.not_visited.remove(self.not_visited[0])
        print('not visited',len(self.not_visited))

        while len(self.not_visited) != 0:
            #print(len(self.not_visited))
            while curr_bin.is_peak == False:
                #Move until finding a peak
                curr_bin = self.uphill_move(curr_bin)
            #If peak not already found, add it to peaks
            if curr_bin.pos not in self.peaks_pos:
                self.peaks_pos.append(curr_bin.pos)
            #Choose a not visted peak to start again
            if len(self.not_visited) != 0:
                new_ind = np.random.randint(0,len(self.not_visited),1)[0]
                hist_ind = self.not_visited[new_ind]
                self.not_visited.remove(hist_ind)
                curr_bin.pos = hist_ind
                curr_bin.is_peak = False
        print('end')
        print(len(self.peaks_pos))
        return len(self.peaks_pos)
    
    def uphill_move(self,curr_bin):
        """
        Make uphill move
        """
        x, y, z = curr_bin.pos
        neigh_coord, neighbors = self.find_neighbors(curr_bin.pos)
        true_neigh = np.delete(neighbors,13) #remove bin itself from neighbors
        
        #Current bin is a peak
        if (self.hist[x,y,z] >= true_neigh).all() and (self.hist[x,y,z] > true_neigh).any():
            curr_bin.is_peak = True
        #All neighbors are equal to current bin    
        elif (self.hist[x,y,z] == neighbors).all():
            curr_bin = self.resolve_draw(curr_bin, neigh_coord)
        #At least one neighbor has greater value than current bin    
        else:
            loc_x, loc_y, loc_z = np.unravel_index(np.argmax(neighbors, axis=None), neighbors.shape)
            curr_bin.pos = neigh_coord[0,loc_x], neigh_coord[1,loc_y], neigh_coord[2,loc_z]
            if curr_bin.pos in self.not_visited:
                self.not_visited.remove(curr_bin.pos)
        return curr_bin
    

    def resolve_draw(self, curr_bin, neigh_coord):
        """
        Resolve conflict when all neighbors are equal to current bin
        """
        for i in neigh_coord[0]:
            for j in neigh_coord[1]:
                for k in neigh_coord[2]:
                    if (i,j,k) in self.not_visited: #We move to a not visited bin
                        curr_bin.pos = (i,j,k)
                        self.not_visited.remove(curr_bin.pos)
                        return curr_bin
        curr_bin.is_peak = True #All neighbors have already been visited so current bin is in a plateau,
        #We label it as peak
        return curr_bin


def get_limits(i, j, n_H, n_W, W_R):
    """
    Returns limits of the window of width W_R centered
    in pixel (i,j) in the integral image and number of
    pixels in the window in the orginal
    """
    #Limits of the window in the integram
    maxi_x = min(n_H, i + 1 + W_R//2)
    maxi_y = min(n_W, j + 1 + W_R//2)
    mini_x = max(0, i - W_R//2)
    mini_y = max(0, j - W_R//2)
    #Number of pixels in the window in the ORIGINAL image (not the integral image)
    N = (maxi_x - mini_x)*(maxi_y - mini_y)
    return int(mini_x), int(mini_y), int(maxi_x), int(maxi_y), int(N)
  
def extract_saliency(img , scale):
    """
    Computes saliency map for a specific scale
    Arguments : 
    image -- numpy array, of shape (n_H, n_W, n_C)
    scale -- integer, representing the scale.
    Outputs :
    saliency_map -- saliency map fo a specific scale, of shape (n_H, n_W)
    """
    n_H, n_W, n_channel = img.shape
    W_R2 = scale
    N2 = W_R2**2
    integral_img = cv2.integral(img)
    saliency_map = np.zeros((n_H,n_W))
    for i in range(n_H):
        for j in range(n_W):
            #Window limits in integral image
            mini_x, mini_y, maxi_x, maxi_y, N = get_limits(i, j, n_H, n_W, W_R2)
            #Sum of window pixels in original image
            tmp = 1.0*(integral_img[mini_x,mini_y,:] + integral_img[maxi_x,maxi_y,:]\
                       - (integral_img[mini_x,maxi_y,:] \
                          + integral_img[maxi_x,mini_y,:]))/N
            #Distance between R1 and R2
            saliency_map[i,j] = np.linalg.norm(img[i,j,:]-np.array(tmp))
    return saliency_map

def mean_saliency(image):
    """
    Compute mean saliency from different scales
    Arguments : 
    image -- numpy array, of shape (n_H, n_W, n_C)
    Outputs :
    M -- mean saliency maps across different scales, of shape (n_H, n_W)
    """
    h, w, n_chan = image.shape
    w2 = min(h,w)
    scales = [w2/2, w2/4, w2/8]
    M = np.zeros((h,w))
    for s in range(3):
        saliency = extract_saliency(image, scales[s])
        M += saliency
    return M

def apply_kmeans(image, K):
    """
    Apply K-means algorithm to the image
    Arguments :
    image -- numpy array, of shape (n_H, n_W, n_C)
    K -- number of classes
    Outputs :
    y_pred -- array containing the class label of each pixel, of shape (n_H, n_W)
    """
    n_H, n_W, n_C = image.shape
    reshaped_image = np.reshape(image,(n_H*n_W, n_C))
    kmeans = KMeans(init='k-means++', n_clusters=K, n_init=10).fit(reshaped_image)
    y_pred = kmeans.predict(reshaped_image)
    return np.reshape(y_pred, (n_H,n_W))

def saliency_means_per_region(image, K):
    """
    Compute mean saliency for each region detected by K means.
    Arguments:
    image -- numpy array, of shape (n_H, n_W, n_C)
    K -- number of classes
    Outputs :
    V -- mean saliency per region, of shape (K,1)
    region_idxes -- array containing class of each pixel, of shape (n_H, n_W)
    """
    M = mean_saliency(image)
    n_H, n_W = M.shape
    region_idxes = apply_kmeans(image, K)
    V = np.zeros((K,))
    for k in range(K):
        N = 0
        for i in range(n_H):
            for j in range(n_W):
                if region_idxes[i,j] == k:
                    V[k] += M[i,j]
                    N += 1
        V[k] = (1.0*V[k])/N
    return V, region_idxes

def segmented_image(image, V , threshold, region_idxes):
    '''
    Function discarding the regions with mean saliency inferior to a threshold.
    Arguments:
    image -- numpy array, of shape (n_H, n_W, n_C)
    V -- mean saliency per region, of shape (K,1)
    region_idxes -- array containing class of each pixel, of shape (n_H, n_W)
    threshold -- value of the threshold
    Outputs :
    segmented_image -- the segmented image, numpy array of shape (n_H, n_W, n_C)
    '''
    n_H, n_W, n_C = image.shape
    test_array = V > threshold
    chosen_classes = [i for i, x in enumerate(test_array) if x]
    segmented_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(n_H):
        for j in range(n_W):
            if region_idxes[i,j] in chosen_classes:
                segmented_image[i,j,:] = image[i,j,:]
    return segmented_image

def main():
    # reading image
    image = cv2.imread(path_to_image)
    # changing color space
    im = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    plt.figure(1)
    plt.title('Original image')
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    # computation of integral image 
    i_im = cv2.integral(im)
    n_bins = 16
    hist = cv2.calcHist([im],[0,1,2],None,[n_bins,n_bins,n_bins],[0,100,-128,127,-128,127])
    hill_climb = Hill_climbing(hist, n_bins)
    # appliying hill climbing algorithm to find k
    k = hill_climb.find_peaks()
    # computation of saliency maps
    V, region_idxes = saliency_means_per_region(im, k)
    threshold = np.median(V) - 0.1
    # segmenting the image
    seg_image = segmented_image(image, V, threshold, region_idxes)
    # color space revert
    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)
    # displaying the image
    plt.figure(2)
    plt.title('Segmented image')
    plt.imshow(seg_image)
    plt.show()
main()