from libc.math cimport abs, sqrt, ceil, floor, log, log2, acos, cos
from numpy cimport ndarray
import numpy as np
from numpy import ones, zeros, int32, float32, uint8, fromstring
from numpy import sort, empty, array, arange, concatenate, searchsorted
from numpy import minimum, maximum,divide, std, mean
from numpy import min as nmin
from numpy import max as nmax
import h5py as h



def non_overlapping(ndarray[int, ndim=2] regions):
    '''
    Given a list of regions which may be overlapping we return a sorted list of regions which aren't overlapping
    arguments:
    - regions: An (N,2) shape array with each row detailing the start and end of a region
    
    outputs:
    - nonoverlapping: An (M,2) shape array where each row details a covered region. The regions in the output array
                      will be sorted by start position and non-overlapping with each other. 
    '''
    cdef int idx, jdx, nr = len(regions)
    cdef ndarray[int, ndim=2] outregs = zeros((nr,2),int32)
    cdef ndarray[int, ndim=2] values = zeros((2*regions.shape[0],2), int32)
    for idx in range(nr):
        values[idx,0] = regions[idx,0]
        values[idx,1] = 1
        values[idx+nr,0] = regions[idx,1]
        values[idx+nr,1] = -1
    cdef ndarray[int, ndim=1] order = array(values[:,0].argsort(), int32)
    
    cdef int total = 0
    cdef int new_point = 0
    cdef int valtype = 0
    cdef int current_start = 0
    cdef int current_end = 0
    cdef int out_idx = 0
    for idx in np.arange(2*nr):
        jdx = order[idx]
        new_point, valtype = values[jdx]
        total += valtype
        if valtype <0:
            current_end = new_point

        if (total == 1) and valtype > 0:
            current_start = new_point
                
        if total == 0:
            outregs[out_idx,0] = current_start
            outregs[out_idx,1] = current_end
            out_idx += 1
            
    return outregs[:out_idx,:]


def pairRegionsIntersection(ndarray[int, ndim=2] pairs,
                            ndarray[int, ndim=2] regions,
                            exclude=False, allow_partial=False,
                            bool_out=False):
    '''
    Given an list of pairs of the form [a,b] where a<b and regions of the form [c,d] we want to return the indices
    (within the pair list) of pairs which overlap with the regions. This will be used to exclude certain regions of the
    genome if needed.
     
    Arguments:
    
    - pairs: (N,2) shape array where each row details a pair.
    - regions: (M,2) shape array where each row details a region.
    - exclude: Boolean. If true, then return the indices of pairs which don't overlap with any of the regions.
    - allow_partial: Boolean. If true then include pair indices which only partially overlap with a given region.
                     Here we consider the 'partial' to apply to the pair. That is, if allow_partial == False then
                     we must satisfy either:
                         - the entire pair is included in a region if exclude == False
                         - The entire pair must not be included in a region if exclude == True
                     If allow_partial == True then we must satisfy:
                         - At least part of the pair touches at least one region if exclude == False
                         - At least part of the pair touches no regions if exclude == True
                      
    Returns:
    
    - indices: The indices of the pairs in the pair array which satisfy the overlapping conditions.
    '''
    cdef int i, j, k, a, b
    cdef int exc = int(exclude)
    cdef int partial = int(allow_partial)
    cdef int bool_o = int(bool_out)
    cdef int ni = 0
    cdef int np = len(pairs)
    cdef int nr = len(regions)
    cdef ndarray[int, ndim=1] indices = zeros(np, int32)
    cdef ndarray[int, ndim=1] indices_out = empty(np, int32)
    cdef ndarray[int, ndim=1] order = array(regions[:,0].argsort(), int32)
    cdef ndarray[int, ndim=2] regs = empty((nr+1,2), int32)
    cdef minpoint = minimum(nmin(pairs), nmin(regions))
    cdef maxpoint = maximum(nmax(pairs), nmax(regions))

    if exc:
        for i in range(nr):
            regs[i,1] = regions[order[i],0]
            regs[i+1,0] = regions[order[i],1]
        regs[0,0] = minpoint - 1
        regs[-1,1] = maxpoint + 1
    else:
        regs = regions
    
    
    for i in range(np):
        for k in range(nr):
            j = order[k]
            if (regs[j,1] > pairs[i,0]) and (pairs[i,1] > regs[j,0]):
                #Pair is at least partially overlapping with the region
                if partial:
                    indices[i] = 1
                    if bool_o:
                        return 1
                    continue
                elif (regs[j,1] >= pairs[i,1]) and (pairs[i,0] >= regs[j,0]):
                    #Pair is entirely containing within the region
                    indices[i] = 1
                    if bool_o:
                        return 1
                    continue
    
    if bool_o:
        return 0
    
    for i in range(np):
        if indices[i] == 1:
            indices_out[ni] = i
            ni +=1
    
    return indices_out[:ni]


def link_parent_and_child_multi_regions(ndarray[int, ndim=3] pairs,
                                        ndarray[int, ndim=3] regions,
                                        int cutoff = -1,
                                        allow_partial=False):
    '''
    Given an list of pairs of the form [a,b] where a<b and regions of the form [c,d] we want to work pairwise to
    return the region indices of the regions which overlap with each pair.
     
    Arguments:
    
    - pairs: (N,2) shape array where each row details a pair.
    - regions: (M,2) shape array where each row details a region.
    - exclude: Boolean. If true, then return the indices of pairs which don't overlap with any of the regions.
    - allow_partial: Boolean. If true then include pair indices which only partially overlap with a given region.
                     Here we consider the 'partial' to apply to the pair. That is, if allow_partial == False then
                     we must satisfy either:
                         - the entire pair is included in a region if exclude == False
                         - The entire pair must not be included in a region if exclude == True
                     If allow_partial == True then we must satisfy:
                         - At least part of the pair touches at least one region if exclude == False
                         - At least part of the pair touches no regions if exclude == True
                      
    Returns:
    
    - out: A maximum of an (N*M,2) shape array where each column details a pair group index and a regions group
           index where than pair group and that region group overlap. This is essentially gonna be in COO
           format
    '''
    cdef int i, j, k, np, nr, nl = 0
    cdef int overlap
    cdef int maxnp = len(pairs[0,:,0]) 
    cdef int maxnr = len(regions[0,:,0])
    cdef int npairgroups = len(pairs)
    cdef int nreggroups = len(regions)
    cdef ndarray[int, ndim=2] out = zeros((npairgroups*nreggroups,2), int32)
    
    
    for i in range(npairgroups):
        np = maxnp
        for k in range(maxnp):
            if pairs[i,k,0] == cutoff:
                np = k
                break
        for j in range(nreggroups):
            nr = maxnr
            for k in range(maxnr):
                if regions[j,k,0] == cutoff:
                    nr = k
                    break

            overlap = link_parent_and_child_regions(pairs[i,:np,:],
                                                    regions[j,:nr,:],
                                                    allow_partial = allow_partial,
                                                    bool_out = True)
            if overlap == 1:
                out[nl,0] = i
                out[nl,1] = j
                nl +=1
    
    return out[:nl,:]
                


def link_parent_and_child_regions(ndarray[int, ndim=2] parent_regs,
                                  ndarray[int, ndim=2] child_regs,
                                  allow_partial=False, 
                                  exclude = False,
                                  bool_out = False):
    '''
    Given some pairs and regions we want to return a COO matrix detailing overlaps between the parent
    and child regions.
    
    Arguments:
    
    - pairs: (N,2) shape array where each row details a pair.
    - regions: (M,2) shape array where each row details a region.
    - exclude: Boolean. If true then we retrurn a COO matrix of regions and pairs which don't satisfy our
               constraint.
    - allow_partial: Boolean. If true then include pair indices which only partially overlap with a given region.
                     Here we consider the 'partial' to apply to the pair. That is, if allow_partial == False then
                     we must satisfy either:
                         - the entire pair is included in a region if exclude == False
                         - The entire pair must not be included in a region if exclude == True
                     If allow_partial == True then we must satisfy:
                         - At least part of the pair touches at least one region if exclude == False
                         - At least part of the pair touches no regions if exclude == True
                      
    Returns:
    
    - out: A maximum of an (N*M,2) shape array where each column details a pair group index and a regions group
           index where than pair group and that region group overlap. This is essentially gonna be in COO
           format
    '''
    cdef int i, j, k
    cdef int exc = int(exclude)
    cdef int partial = int(allow_partial)
    cdef int np = len(parent_regs)
    cdef int nr = len(child_regs)
    cdef int bool_o = int(bool_out)
    cdef int nl = 0
    cdef ndarray[int, ndim=2] out = empty((np*nr,2), int32)

    
    for i in range(np):
        for j in range(nr):
            if (child_regs[j,1] > parent_regs[i,0]) and (parent_regs[i,1] > child_regs[j,0]):
                #Pair is at least partially overlapping with the region
                if partial:
                    if bool_o:
                        return 1
                    out[nl,0] = i
                    out[nl,1] = j
                    nl += 1
                    continue
                elif (child_regs[j,1] <= parent_regs[i,1]) and (child_regs[j,0] >= parent_regs[i,0]):
                    if exc == 0:
                        if bool_o:
                            return 1
                        out[nl,0] = i
                        out[nl,1] = j
                        nl += 1
            elif exc == 1:
                out[nl,0] = i
                out[nl,1] = j
                nl+=1
    if bool_o:
        return 0
    
    return out[:nl,:]
                

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
    
    pairs = non_overlapping(pairs)
    regions = non_overlapping(regions)
    cdef int i, j, k
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
    cdef ndarray[int, ndim=1] current_pair = empty(2, int32)
    
    for i in range(np):
        current_pair = pairs[i,:]
        
        if current_pair[1] < regions[order[0],0]:
            if exc:
                overlap_bounds[tnio,:] = current_pair
                tnio += 1
                
            continue
        
        if current_pair[0] > regions[order[nr-1],1]:
            if exc:
                overlap_bounds[tnio,:] = current_pair
                tnio += 1
                
            continue
            
        for k in range(nr):
            j = order[k]
            if regions[j,0] < current_pair[1] and regions[j,1] > current_pair[0]:
                #There is an overlap between this region and this pair chunk
                if regions[j,0] <= current_pair[0] and current_pair[0] < regions[j,1]:
                    #Start of the current pair chunk is in the region
                    if regions[j,1] >= current_pair[1]:
                        #Pair chunk entirely contained with a region 
                        if exc:
                            #If exclude then we know that this
                            #pair can't be included - it is contained within a 
                            #region so break the current loop and move onto the 
                            #next pair
                            break
                        else:
                            #If include then we want to add this pair chunk to the 
                            #list of output pair chunks
                            overlap_bounds[tnio] = current_pair
                            tnio += 1
                            continue
                    else:
                        #Start of pair chunk is in the region, end isnt
                        if exc:
                            #If exclude then we just shave off the bit of this
                            #pair that touches the reegion and continue to the 
                            #next region
                            current_pair[0] = regions[j,1]
                        else:
                            #If include then we add this overlap to the 
                            #output and shave off the bit of the pair that overlaps
                            #with the region and then continue on to the next region
                            overlap_bounds[tnio,0] = current_pair[0]
                            overlap_bounds[tnio,1] = regions[j,1]
                            
                            tnio += 1
                            
                            current_pair[0] = regions[j,1]
                            continue
                if regions[j,0] > current_pair[0]:
                    #Start of the current pair chunk is lower than the start of
                    #the region
                    if regions[j,1] < current_pair[1]:
                        #End of the current pair chunk is higher than the start of the
                        #regions - i.e. region entirely contained within current
                        #pair chunk
                        if exc:
                            #Since the regions are sorted we know that the lower portion
                            #of the pair which is cutoff can't overlap with any other 
                            #region so we add it to the output
                            overlap_bounds[tnio,0] = current_pair[0]
                            overlap_bounds[tnio,1] = regions[j,0]
                            tnio += 1
                            
                            #We then shave the current pair to only include the bit
                            #which didn't overlap with the current region above
                            current_pair[0] = regions[j,1]
                        else:
                            #We include the overlapping region
                            overlap_bounds[tnio,:] = regions[j,:]
                            tnio += 1
                            
                            #Then shave the current pair
                            current_pair[0] = regions[j,1]
                    else:
                        #Current pair end must overlap with the region
                        if exc:
                            overlap_bounds[tnio,0] = current_pair[0]
                            overlap_bounds[tnio,1] = regions[j,0]
                            tnio +=1
                            #Since the regions are sorted we know that
                            #we've already got all the info out
                            #of this pair
                            current_pair[0] = regions[j,0]
                            break
                        else:
                            overlap_bounds[tnio,0] = regions[j,0]
                            overlap_bounds[tnio,1] = current_pair[1]
                            tnio += 1
                            #Since the regions are sorted we know that
                            #there won't be any more of this pair that
                            #is overlapping any regions so continue
                            #to the next pair
                            current_pair[0] = regions[j,0]
                            break
            elif exc and regions[j,0] > current_pair[1]:
                #If we find ourself with a pair whose end is less
                #than the current region start and we're also excluding
                #then we can just go ahead and input this pair
                overlap_bounds[tnio,:] = current_pair 
                tnio += 1
                break
                    
          
    return overlap_bounds[:tnio]

def binrvps_constantbins(ndarray[int, ndim=2] regions,
                    ndarray[double, ndim=1] values,
                    int binSize=1000,
                    int start=0,
                    int end=-1,
                    double dataMax=0.0,
                    double scale=1.0,
                    double threshold=0.0):
    
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
               maximum absolute binned value is greater than dataMax then this is used as the normalising 
               constant instead.
    - scale: Factor by which to scale all data by after binning.
    - threshold: Minimum region-value to use for binning. If a given region has value < threshold then this
                 region is excluded from the binning. 
                 
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
        end = regions.max()

    e = int((end-start-1)/binSize)
    #Number of bins Must be at least one if e-s < binSize
    nBins = e+1

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
        
        if p1 == p2:
            continue
        
        #If the start of the pair is greater than or equal to our
        #end then we dont care about it
        if end <= p1:
            continue

        if start > p2:
            continue
            
        if p1 < start:
            p1 = start
    
        if p2 >= end:
            p2 = end

        b1 = int32((p1-start)/binSize)
        b2 = int32((p2-start-1)/binSize)
    
        if b1 == b2:
            r = <double> (p2-p1)
            hist[b1] += v*r
        else:
            for b3 in range(b1, b2+1):
                if b3 > nBins-1:
                    break

                if b3 * binSize < p1 - start:
                    f = <double> (start+(b3+1)*binSize - p1)

                elif (b3+1) * binSize > p2:
                    f = <double> (p2 - b3*binSize - start) 
                else:
                    f = <double> binSize
                hist[b3] += v * f

    hist /= binSize
    if nBins > 1:
        hist[nBins-1] *= binSize/(end - start - binSize*(nBins-1))
    else:
        hist[0] *= binSize/(end-start)
    
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

def binrvps(ndarray[int, ndim=2] regions,
           ndarray[double, ndim=1] values,
           ndarray[int, ndim=1] bins,
           double dataMax=0.0,
           double scale=1.0,
           double threshold=0.0):
    '''
    Given some (N,2) array where each row is some interval [a,b] with a<b and given some values over each
    of those intervals, bin that data track into a length K vector of regular intervals each of some given 
    bin size. 
    
    Arguments:
    
    - regions: (M,2) shape array where each row details a region.
    - values: (M,) shape array detailing the data track value for each of the given regions.
    - bins: (M+1) shape array where M is the number of bins and bins[i] is the start index
            of bin[i] for i < M. bins[M] is the end of the final bin.
    - dataMax: If dataMax is set to some non-zero value then dataMax is used as a normalising constant for the
               binned data track. That is, after initial binning the maximum and minimum data values across
               the bins are computed. If dataMax is greater than the maximum absolute data track value then
               this is used as a normalising constant i.e. all binned values are divided by dataMax. If the 
               maximum absolute binned value is greater than dataMax then this is used as the normalising 
               constant instead.
    - scale: Factor by which to scale all data by after binning.
    - threshold: Minimum region-value to use for binning. If a given region has value < threshold then this
                 region is excluded from the binning. 
                 
    Returns:
    
    - hist: A histogram of the data (the binned data track). This will
            be a ((end-start)/binSize,) shape array
    '''
    cdef int i, j, p1, p2, b1, b2, b3, s, e, start, end
    cdef int nBins, n = len(values)
    cdef double f, r, v, vMin, vMax
  
    if len(regions) != n:
        data = (len(regions), n)
        raise Exception('Number of regions (%d) does not match number of values (%d)' % data) 
  
  
    start = bins[0]
    end = bins[-1]
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
    
        if end <= p1:
            continue
    
        if start > p2:
            continue  
    
        if start > p1:
            p1 = start
        
        if end <= p2:
            p2 = end

        b1 = int32(np.where(bins<=p1)[0][-1])
        b2 = int32(np.where(bins<=p2)[0][-1])
    
        r = <double> (p2-p1)
        if b1 == b2:
            hist[b1] += v*r

        else:
            for b3 in range(b1, b2+1):
                if b3 < 0:
                    continue
        
                if b3 >= nBins:
                    break  
      
                if bins[b3] < p1:
                    f = <double> (bins[b3+1] - p1) 
        
                elif bins[b3+1] > p2:
                    f = <double> (p2 - bins[b3]) 
        
                else:
                    f = bins[b3+1] - bins[b3]
      
                hist[b3] += v * f
    
    for i in range(nBins):
        if bSizes[i] != 0:
            hist[i] /= bSizes[i]

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
            
    hist *= scale 
        
    return hist



def rvps_to_rvps(ndarray[int, ndim=2] regions,
                 ndarray[double, ndim=1] values,
                 ndarray[int, ndim=2] newregs,
                 double dataMax=0.0,
                 double scale=1.0,
                 double threshold=0.0,
                 int stats_type = 0,
                 int stats_scale = 1
                ):
    '''
    Given some (N,2) array where each row is some interval [a,b] with a<b and given some values over each
    of those intervals, bin that data track into a length K vector of regular intervals each of some given 
    bin size. 
    
    Arguments:
    
    - regions: (M,2) shape array where each row details a region.
    - values: (M,) shape array detailing the data track value for each of the given regions.
    - newregs: (N,2) shape array detailing some new regions which we want to assign values to based on our
               region value pairs. 
    - dataMax: If dataMax is set to some non-zero value then dataMax is used as a normalising constant for the
               binned data track. That is, after initial binning the maximum and minimum data values across
               the bins are computed. If dataMax is greater than the maximum absolute data track value then
               this is used as a normalising constant i.e. all binned values are divided by dataMax. If the 
               maximum absolute binned value is greater than dataMax then this is used as the normalising 
               constant instead.
    - scale: Factor by which to scale all data by after binning.
    - threshold: Minimum region-value to use for binning. If a given region has value < threshold then this
                 region is excluded from the binning.
    - stats_type: determines what statistic to determine for each bin from:
                  0 - per basepair mean
                  1 - per basepair sum
                  2 - per stats_scale*basepair min
                  3 - per stats_scale*basepair max
                  4 - per stats_scale*basepair standard deviation
                  5 - coverage
                  6 - per stats_scale*region mean
                  7 - per stats_scale*region standard deviation
                  8 - per stats_scale*region min
    - stats_scale: Since calculating statistics such as the per basepair max requires calculating the binning
                   for each basepair in every region, with very large output regions this can be very memory
                   intensive. If this is an issue we can set the stats scale to be larger - e.g. calculating
                   the maximum value over each 100 basepair region.
                 
    Returns:
    
    - hist: A histogram of the data (the binned data track). This will
            be a ((end-start)/binSize,) shape array
    '''
    cdef int nBins, n = len(values)
    cdef int p1, p2
    cdef ndarray[int, ndim=1] reg = zeros(2,int32)
  
    #Require each region to be associated with a value
    if len(regions) != n:
        data = (len(regions), n)
        raise Exception('Number of regions (%d) does not match number of values (%d)' % data) 
  
    #Number of output regions
    nBins = newregs.shape[0]
    #Require a float output for each output region
    cdef ndarray[double, ndim=1] hist = zeros(nBins, float)
    
    if stats_type == 0:
        for i in range(nBins):
            reg = newregs[i,:]
            p1 = reg[0]
            p2 = reg[1]
            if reg[1] < reg[0]:
                p1 = reg[1]
                p2 = reg[0]
                
            reg[0] = p1
            reg[1] = p2
            hist[i] = binrvps(regions,
                              values,
                              reg,
                              dataMax= dataMax,
                              scale=scale,
                              threshold=threshold)[0] 
    
        return hist
    elif stats_type == 1:
        for i in range(nBins):
            reg = newregs[i,:]
            p1 = reg[0]
            p2 = reg[1]
            if reg[1] < reg[0]:
                p1 = reg[1]
                p2 = reg[0]
                
            reg[0] = p1
            reg[1] = p2
            bSize = reg[1]-reg[0]
            hist[i] = bSize*binrvps(regions,
                                    values,
                                    reg,
                                    dataMax= dataMax,
                                    scale=scale,
                                    threshold=threshold)[0] 
    
        return hist
    elif stats_type == 5:
        for i in range(nBins):
            reg = newregs[i,:]
            p1 = reg[0]
            p2 = reg[1]
            if reg[1] < reg[0]:
                p1 = reg[1]
                p2 = reg[0]
                
            reg[0] = p1
            reg[1] = p2
            
            bSize = reg[1]-reg[0]
            hist[i] = bSize*binrvps(regions,
                                    ones(n,float),
                                    reg,
                                    dataMax= dataMax,
                                    scale=scale,
                                    threshold=threshold)[0] 
        
        return hist
    elif stats_type == 2:
        for i in range(nBins):
            reg = newregs[i,:]
            p1 = reg[0]
            p2 = reg[1]
            if reg[1] < reg[0]:
                p1 = reg[1]
                p2 = reg[0]
            
            hist[i] = stats_scale*nmin(binrvps_constantbins(regions,
                                                              values,
                                                              binSize=stats_scale,
                                                              start=p1,
                                                              end=p2,
                                                              dataMax=dataMax,
                                                              scale=scale,
                                                              threshold= threshold))
        return hist
    elif stats_type == 3:
        for i in range(nBins):
            reg = newregs[i,:]
            p1 = reg[0]
            p2 = reg[1]
            if reg[1] < reg[0]:
                p1 = reg[1]
                p2 = reg[0]
                
            hist[i] = stats_scale*nmax(binrvps_constantbins(regions,
                                                            values,
                                                            binSize=stats_scale,
                                                            start=p1,
                                                            end=p2,
                                                            dataMax=dataMax,
                                                            scale=scale,
                                                            threshold= threshold))
        return hist
    elif stats_type == 4:
        for i in range(nBins):
            reg = newregs[i,:]
            p1 = reg[0]
            p2 = reg[1]
            if reg[1] < reg[0]:
                p1 = reg[1]
                p2 = reg[0]
                
            hist[i] = stats_scale*std(binrvps_constantbins(regions,
                                                            values,
                                                            binSize=stats_scale,
                                                            start=p1,
                                                            end=p2,
                                                            dataMax=dataMax,
                                                            scale=scale,
                                                            threshold= threshold))
        return hist
    
    cdef ndarray[double, ndim=1] coverage
    cdef ndarray[double, ndim=1] interim_vals
    
    if stats_type == 6:
        for i in range(nBins):
            reg = newregs[i,:]
            p1 = reg[0]
            p2 = reg[1]
            if reg[1] < reg[0]:
                p1 = reg[1]
                p2 = reg[0]
            interim_vals = binrvps_constantbins(regions,
                                            values,
                                            binSize=stats_scale,
                                            start=p1,
                                            end=p2,
                                            dataMax=dataMax,
                                            scale=scale,
                                            threshold= threshold)
            coverage = binrvps_constantbins(regions,
                                            ones(n,float),
                                            binSize=stats_scale,
                                            start=p1,
                                            end=p2,
                                            dataMax=dataMax,
                                            scale=scale,
                                            threshold= threshold)
            
            if np.sum(coverage) > 0:
                hist[i] = mean(divide(interim_vals[coverage>0], coverage[coverage>0]))
            else:
                hist[i] = 0
        return hist
    elif stats_type == 7:
        for i in range(nBins):
            reg = newregs[i,:]
            p1 = reg[0]
            p2 = reg[1]
            if reg[1] < reg[0]:
                p1 = reg[1]
                p2 = reg[0]

            coverage = binrvps_constantbins(regions,
                                            ones(n,float),
                                            binSize=stats_scale,
                                            start=p1,
                                            end=p2,
                                            dataMax=dataMax,
                                            scale=scale,
                                            threshold= threshold)
            interim_vals = binrvps_constantbins(regions,
                                            values,
                                            binSize=stats_scale,
                                            start=p1,
                                            end=p2,
                                            dataMax=dataMax,
                                            scale=scale,
                                            threshold= threshold)
            if np.sum(coverage) > 0:
                hist[i] = std(divide(interim_vals[coverage>0], coverage[coverage>0]))
            else:
                hist[i] = 0
        return hist
    elif stats_type == 8:
        for i in range(nBins):
            reg = newregs[i,:]
            p1 = reg[0]
            p2 = reg[1]
            if reg[1] < reg[0]:
                p1 = reg[1]
                p2 = reg[0]

            coverage = binrvps_constantbins(regions,
                                            ones(n,float),
                                            binSize=stats_scale,
                                            start=p1,
                                            end=p2,
                                            dataMax=dataMax,
                                            scale=scale,
                                            threshold= threshold)
            interim_vals = binrvps_constantbins(regions,
                                            values,
                                            binSize=stats_scale,
                                            start=p1,
                                            end=p2,
                                            dataMax=dataMax,
                                            scale=scale,
                                            threshold= threshold)
            if np.sum(coverage) > 0:
                hist[i] = nmin(divide(interim_vals[coverage>0], coverage[coverage>0]))
            else:
                hist[i] = 0
        return hist

    
def binrvps_multi_interval(ndarray[int, ndim=2] regions,
                           ndarray[double, ndim=1] values,
                           ndarray[int, ndim=1] mids,
                           int halfwidth,
                           int binSize,
                           double dataMax=0.0,
                           double scale=1.0,
                           double threshold=0.0): 
    '''
    Given some regions and values as well as the middles of some genomic region, evaluate the rvp function
    in some region each side of the midpoints.
    Argument:
    - regions: (M,2) shape array where each row details a region.
    - values: (M,) shape array detailing the data track value for each of the given regions.
    - mids: (N,) shape array detailing the middles of the regions we want to get the datatrack value in
    - halfwidth: The buffer to put on either side of the midpoint. Integer.
    - binSize: The resolution of the function.
    - dataMax: If dataMax is set to some non-zero value then dataMax is used as a normalising constant for the
               binned data track. That is, after initial binning the maximum and minimum data values across
               the bins are computed. If dataMax is greater than the maximum absolute data track value then
               this is used as a normalising constant i.e. all binned values are divided by dataMax. If the 
               maximum absolute binned value is greater than dataMax then this is used as the normalising 
               constant instead.
    - scale: Factor by which to scale all data by after binning.
    - threshold: Minimum region-value to use for binning. If a given region has value < threshold then this
                 region is excluded from the binning.
    
    '''
    cdef int i,e,s,n = len(mids)
    cdef ndarray[int, ndim=1] nBins = zeros(n, int32)
    for i in range(n):
        s = int32((mids[i]-halfwidth)/binSize)
        #the end is an open interval - we don't want to include basepairs overlapping with the end
        #input
        e = int32((mids[i]+halfwidth-1)/binSize)
        #Number of bins Must be at least one if e-s < binSize
        nBins[i] = e-s+1
    cdef ndarray[double, ndim=2] hist = zeros((n,int32(nmax(nBins))))
    for i in range(n):
        hist[i,:nBins[i]] = binrvps_constantbins(regions,
                                         values,
                                         binSize=binSize,
                                         start=mids[i]-halfwidth,
                                         end=mids[i]+halfwidth,
                                         dataMax=dataMax,
                                         scale=scale,
                                         threshold= threshold)
        
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
