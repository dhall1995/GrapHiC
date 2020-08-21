
from libc.math cimport abs, sqrt, ceil, floor, log, log2, acos, cos
from numpy cimport ndarray
import numpy as np
from numpy import ones, zeros, int32, float32, uint8, fromstring
from numpy import sort, empty, array, arange, concatenate, searchsorted
import h5py as h


def pairRegionsIntersection(ndarray[int, ndim=2] pairs,
                            ndarray[int, ndim=2] regions,
                            exclude=False, allow_partial=False,
                            region_indices=False):
    '''
    Given an list of pairs of the form [a,b] where a<b and regions of the form [c,d] we want to return the indices
    (within the pair list) of pairs which overlap with the regions. This will be used to exclude certain regions of the
    genome if needed.
     
    Arguments:
    
    - pairs: (N,2) shape array where each row details a pair.
    - regions: (M,2) shape array where each row details a region.
    - exclude: Boolean. If true, then return the indices of pairs which don't overlap with any of the regions.
    - allow_partial: Boolean. If true then include pair indices which only partially overlap with a given region.
    - region_indices: Boolean. If true then also return the indices of regions in the regions list which satisfy the
                      conditions.
                      
    Returns:
    
    - indices: The indices of the pairs in the pair array which satisfy the overlapping conditions.
    - regions: If region_indices == True then the function returns the indices of the pairs in the region array which 
               satisfy the overlapping conditions.
               
    '''
    cdef int i, j, k, a, b
    cdef int exc = int(exclude)
    cdef int partial = int(allow_partial)
    cdef int ni = 0
    cdef int np = len(pairs)
    cdef int nr = len(regions)
    cdef ndarray[int, ndim=1] indices = empty(np, int32)
    cdef ndarray[int, ndim=1] indices_reg = empty(np, int32)
    cdef ndarray[int, ndim=1] order = array(regions[:,0].argsort(), int32)

    for i in range(np):

        if pairs[i,1] < regions[order[0],0]:
            if exc:
                indices[ni] = i
                ni += 1

            continue

        if pairs[i,0] < regions[order[0],0]:
            if exc and partial:
                indices[ni] = i
                ni += 1

            continue

        a = 0
        b = 0

        for k in range(nr):
            j = order[k]
            #print i, j, k

            if (regions[j,0] <= pairs[i,0]) and (pairs[i,0] <= regions[j,1]):
                a = 1

            if (regions[j,0] <= pairs[i,1]) and (pairs[i,1] <= regions[j,1]):
                b = 1

            if (pairs[i, 0] < regions[j, 0]) and (pairs[i, 1] < regions[j, 0]):
                break

            if partial & (a | b):
                break
            elif a & b:
                break

        if partial:
            if exc and not (a & b):
                indices[ni] = i
                indices_reg[ni] = j
                ni += 1

            elif a | b:
                indices[ni] = i
                indices_reg[ni] = j
                ni += 1

        else:
            if exc and not (a | b):
                indices[ni] = i
                indices_reg[ni] = j
                ni += 1

            elif a & b:
                indices[ni] = i
                indices_reg[ni] = j
                ni += 1


    if region_indices:
        return indices[:ni], indices_reg[:ni]

    else:
        return indices[:ni]

    
def pairRegionsOverlap(ndarray[int, ndim=2] pairs,
                       ndarray[int, ndim=2] regions,
                       exclude=False):

    '''
    Given some (N,2) shape array of pairs a,b with a<b and an (M,2) shape array of regions c,d, with c<d, return a new
    (R,2) shape array of overlaps e,f with e<f such that each overlap is the intersection of the sets P and R where P
    is the union of the pairs and R is the union of the regions. 
    
    Arguments:
    
    - pairs: (N,2) shape array where each row details a pair.
    - regions: (M,2) shape array where each row details a region.
    - exclude: Boolean. If True then the function returns a new set of overlaps composed of the overlaps in the set
               P union ~R. 
               
    Returns:
    - overlap_bounds: An (R,2) array detailing the set of intervals making up the overlaps between the pairs and
                      regions
    '''
    cdef int i, j, k, a, b
    cdef int exc = int(exclude)
    #totaloverlaps index counter
    cdef int tnio = 0
    #number of pairs
    cdef int np = len(pairs)
    #number of regions
    cdef int nr = len(regions)
    #order array for the regions
    cdef ndarray[int, ndim=1] order = array(regions[:,0].argsort(), int32)
    #Output regions array (maximum number of possible regions would be np + nr -1
    cdef ndarray[int, ndim=2] overlap_bounds = empty(((np + nr),2), int32)
    cdef ndarray[int, ndim=1] currentpair = empty(2, int32)
    for i in range(np):
        currentpair = pairs[i,:]
        
        if pairs[i,1] < regions[order[0],0]:
            if exc:
                overlap_bounds[tnio,:] = pairs[i,:]
                tnio += 1
                
            continue
        
        if pairs[i,0] > regions[order[nr-1],1]:
            if exc:
                overlap_bounds[tnio,:] = pairs[i,:]
                tnio += 1
                
            continue

        a = 0
        b = 0

        for k in range(nr):
            j = order[k]
            #print i, j, k
            if (regions[j,0] <= currentpair[0]) and (currentpair[0] <= regions[j,1]):
                if (regions[j,1] >= currentpair[i,1]):
                    if exc:
                        continue
                    else:
                        overlap_bounds[tnio,:] = currentpair
                        tnio += 1
                        
                        break
                else:
                    if exc:
                        currentpair[0] = regions[j,1]
                    else:
                        overlap_bounds[tnio,0] = currentpair[0]
                        overlap_bounds[tnio,1] = regions[j,1]
                        tnio += 1
            elif (regions[j,0] > currentpair[0]) and (currentpair[1] > regions[j,0]):
                if (regions[j,1] >= currentpair[i,1]):
                    if exc:
                        overlap_bounds[tnio,0] = currentpair[0]
                        overlap_bounds[tnio,1] = regions[j,0]
                        tnio += 1
                        
                        break
                    else:
                        overlap_bounds[tnio,0] = regions[j,0]
                        overlap_bounds[tnio,1] = currentpair[-1]
                        tnio += 1
                        break
                else:
                    if exc:
                        overlap_bounds[tnio,0] = currentpair[0]
                        overlap_bounds[tnio,1] = regions[j,0]
                        
                        tnio +=1
                    else:
                        overlap_bounds[tnio,:] = regions[j,:]
                        tnio +=1
                        
                    currentpair[0] = regions[j,1]
           
           
    return overlap_bounds[:tnio]


def regionBinValues(ndarray[int, ndim=2] regions,
                    ndarray[double, ndim=1] values,
                    int binSize=1000,
                    int start=0,
                    int end=-1,
                    double dataMax=0.0,
                    double scale=1.0,
                    double threshold=0.0,
                    int norm_value_by_region = 1
                   ):
    
    '''
    Given some (N,2) array where each row is some interval [a,b] with a<b and given some values over each
    of those intervals, bin that data track into a length K vector of regular intervals each of some given 
    bin size. 
    
    Arguments:
    
    - regions: (M,2) shape array where each row details a region.
    - values: (M,) shape array detailing the data track value for each of the given regions.
    - binSize: Integer. The size of each regular bin.
    - start: The minimum possible bin. All regions [a,b] with b < start are excluded from the binning. Regions
             with a < start < b are clipped - the proportion of the bin overlapping with the allowed interval
             defined by [start,end] is multiplied by the value of the original region. 
    - end: The maximum possible bin. All regions [a,b] with end < a are excluded from the binning. Regions
           with a < end < b are clipped - the proportion of the bin overlapping with the allowed interval
           defined by [start,end] is multiplied by the value of the original region.
    - dataMax: If dataMax is set to some non-zero value then dataMax is used as a normalising constant for the
               binned data track. That is, after initial binning the maximum and minimum data values across
               the bins are computed. If dataMax is greater than the maximum absolute data track value then
               this is used as a normalising constant i.e. all binned values are divided by dataMax. If the 
               maximum absolute binned value is greater than dataMax then this is used ast the normalising 
               constant instead.
    - scale: Factor by which to scale all data by after binning.
    - threshold: Maximum region-value to use for binning. If a given region has value > threshold then this
                 region is excluded from the binning. 
    - norm_value_by_region: Each region is associated with a value regardless of region width. 
                            If norm_values_by_region then the value associated with each region
                            is multiplied by {region_width}/{binSize}. Essentially this means
                            that a 10 basepair region with value 10 is treated the same as a 
                            100 basepair region with value 1 if both regions fall entirely within
                            the same bin. Think of this as enforcing an 'area under the curve'
                            style binning as opposed to weighting all regions equally.
                 
    Returns:
    
    - hist: A histogram of the data (the binned data track) with each histogram bin of size binSize. This will
            be a ((end-start)/binSize,) shape array
    '''

    cdef int i, p1, p2, b1, b2, b3, s, e
    cdef int nBins, n = len(values)
    cdef double f, r, v, vMin, vMax

    if len(regions) != n:
        data = (len(regions), n)
        raise Exception('Number of regions (%d) does not match number of values (%d)' % data)

    if end < 0:
        end = binSize * int32(regions.max() / binSize)

    s = int32(start/binSize)
    e = int32(end/binSize)
    nBins = e-s

    cdef ndarray[double, ndim=1] hist = zeros(nBins, float)

    for i in range(n):

        v = values[i]
        if abs(v) < threshold:
            continue

        if regions[i,0] > regions[i,1]:
            p1 = regions[i,1]
            p2 = regions[i,0]

        else:
            p1 = regions[i,0]
            p2 = regions[i,1]
            
        if norm_value_by_region == 1:
            r = <double> p2 - p1
            v = v*r/binSize

        if end < p1:
            continue

        if start > p2:
            continue

        b1 = int32(p1 / binSize)
        b2 = int32(p2 / binSize)

        if b1 == b2:
            if b1 < s:
                continue

            if b1 > e:
                continue

            hist[b1-s] += v

        else:
            r = <double> (p2-p1)

            for b3 in range(b1, b2+1):
                if b3 < s:
                    continue

                if b3 >= e:
                    break

                if b3 * binSize < p1:
                    f = <double> ((b3+1)*binSize - p1) / r

                elif (b3+1) * binSize > p2:
                    f = <double> (p2 - b3*binSize) / r

                else:
                    f = 1.0

                hist[b3-s] += v * f

    if dataMax != 0.0:
        vMin = hist[0]
        vMax = hist[0]

        for i in range(1, nBins):
            if hist[i] < vMin:
                vMin = hist[i]

            elif hist[i] > vMax:
                vMax = hist[i]

        vMax = max(abs(vMin), vMax, dataMax)

        if vMax > 0.0:
            for i in range(0, nBins):
                hist[i] = hist[i]/vMax

    for i in range(0, nBins):
        hist[i] = hist[i] * scale

    return hist



def NewregionBinValues(ndarray[int, ndim=2] regions,
                    ndarray[double, ndim=1] values,
                    ndarray[int, ndim=1] bins,
                    double dataMax=0.0,
                    double scale=1.0,
                    double threshold=0.0):
    '''
    TEST FUNCTION - CURRENTLY UNUSED
    
    New version of regionBinValues which is being designed to take arbitrary binnings - i.e. non-constant
    bin sizes. This hasn't currently been tested.
    '''
    cdef int i, j, p1, p2, b1, b2, b3, s, e, start, end
    cdef int nBins, n = len(values)
    cdef double f, r, v, vMin, vMax
  
    if len(regions) != n:
        data = (len(regions), n)
        raise Exception('Number of regions (%d) does not match number of values (%d)' % data) 
  
  
    start = bins[0]
    end = bins[-1]
    s = 0
    e = len(bins) - 1
    nBins = len(bins) - 1
  
    # Set up our histograms and create a bin sizes array to allow us to later calculate some 
    # scaled value for a datatrack across different binsizes
    cdef ndarray[double, ndim=1] hist = zeros(nBins, float)
    cdef ndarray[double, ndim=1] bSizes = zeros(nBins, float)
    
    for j in range(nBins):
        bSizes[j] = abs(bins[j+1] - bins[j])
  
    for i in range(n):
    
        v = values[i]
        if abs(v) < threshold:
            continue
    
        if regions[i,0] > regions[i,1]:
            p1 = regions[i,1] 
            p2 = regions[i,0]
    
        else:
            p1 = regions[i,0]
            p2 = regions[i,1]
    
        if end < p1:
            continue
    
        if start > p2:
            continue  
    
        if start > p1:
            b1 = -1
        else:
            b1 = int32(np.where(bins<p1)[0][-1])
        
        if end < p2:
            b2 = e + 1
        else:
            b2 = int32(np.where(bins<p2)[0][-1])
    
        if b1 == b2:
            if b1 < s:
                continue
      
            if b1 > e:
                continue
        
            hist[b1-s] += v

        else:
            r = <double> (p2-p1)
    
            for b3 in range(b1, b2+1):
                if b3 < s:
                    continue
        
                if b3 >= e:
                    break  
      
                if bins[b3] < p1:
                    f = <double> (bins[b3+1] - p1) / r 
        
                elif bins[b3+1] > p2:
                    f = <double> (p2 - bins[b3]) / r 
        
                else:
                    f = 1.0
      
                hist[b3-s] += v * f
  
    for i in range(nBins):
        if bSizes[i] > 0:
            hist[i] = hist[i]/bSizes[i]
    
    vMin = hist[0]
    vMax = hist[0]
    
    for i in range(1, nBins):
        if hist[i] < vMin:
            vMin = hist[i]
      
        elif hist[i] > vMax:
            vMax = hist[i]
    
    vMax = max(abs(vMin), vMax, dataMax)

    if vMax > 0.0:
        for i in range(0, nBins):
            hist[i] = hist[i]/vMax
  
    for i in range(0, nBins):  
        hist[i] = hist[i] * scale  
  
    return hist


# Given a list of contacts of the form [pos1, pos2, n_contacts] we want to place them into a binned matrix where the
# binned indices are offset by wherever the chromosome starts. Can choose to make the matrix symmetric or not or whether
# to also include transposed values in the matrix for the contact.
def binContacts(ndarray[int, ndim=2] contacts,
                ndarray[int, ndim=2] binMatrix,
                int offsetA, int offsetB, int binSize=100000,
                int symm=0, int transpose=0):
    '''
    Given a list of contacts between chromosomes A and B of the form [pos1, pos2, n_contacts] we want to place them into a
    binned matrix where the binned indices are offset by wherever the chromosome starts. Can choose to make the matrix
    symmetric or not or whether to also include transposed values in the matrix for the contact.
    
    Arguments:
    
    - contacts: (N,3) shape array of the form [p1,p2,n_contacts] detailing experimental contacts from a Hi-C experiment.
                p1 and p2 are positions in basepairs while n_contacts is the total number of experimental contacts
                between the two positions.
    - binMatrix: (P,Q) shape contact matrix where P is the number of bins in chromosome A while Q is the number of bins
                 in chromosome B. 
    - offsetA: The starting index in basepairs for chromosome A in the binned contact matrix. For example, if the
               centromeres of chromosomes are not used then the starting contact matrix index will correspond to basepair
               300kb.
    - offsetB: The starting index in basepairs for chromosome B in the binned contact matrix. For example, if the
               centromeres of chromosomes are not used then the starting contact matrix index will correspond to basepair
               300kb.
    - binSize: Integer. The size (in basepairs) of each regular bin.
    - symm: If the binned matrix is symmetric (i.e. if chromosome A == chromosome B) then create a symmetric matrix
    - transpose: Transpose the resulting matrix.
    '''
    cdef int i, a, b
    cdef int n, m, nCont = len(contacts[0])

    n = len(binMatrix)
    m = len(binMatrix[0])

    for i in range(nCont):
        a = int32((contacts[0,i]-offsetA)/binSize)
        b = int32((contacts[1,i]-offsetB)/binSize)
        if transpose:
            a, b = b, a

        if (0 <= a < n) and (0 <= b < m):
            binMatrix[a,b] += contacts[2,i]

        if symm and (a != b):
            binMatrix[b,a] += contacts[2,i]

    return binMatrix


def Var_binContacts(ndarray[int, ndim=2] contacts,
               ndarray[int,ndim=2] binMatrix,
               ndarray[int,ndim=1] bin1,
               ndarray[int,ndim=1] bin2,
               int symm = 0, int transpose = 0):
    '''
    TEST FUNCTION - CURRENTLY UNUSED
    
    New version of binContacts which is being designed to take arbitrary binnings - i.e. non-constant
    bin sizes. This hasn't currently been tested.
    '''
    cdef int i, a, b, j, k
    cdef int n, m, nCont = len(contacts[0])
    
    n = len(bin1)
    m = len(bin2)
    
    for i in range(nCont):
        if contacts[0,i] < bin1[0] or contacts[0,i] > bin1[-1]:
            continue
        
        if contacts[1,i] < bin2[0] or contacts[1,i] > bin2[-1]:
            continue
            
        for j in range(n-1):
            if contacts[0,i] > bin1[j] and contacts[0,i] < bin1[j+1]:
                a = j
            else:
                a = n-1
                
        for k in range(m-1):
            if contacts[1,i] > bin1[k] and contacts[1,i] < bin1[k+1]:
                b = k
            else:
                b = m-1
        
        if transpose:
            a, b = b, a
 
        if (0 <= a < n) and (0 <= b < m):
            binMatrix[a,b] += contacts[2,i]
 
        if symm and (a != b):
            binMatrix[b,a] += contacts[2,i]
  
    return binMatrix
