from dtrack_utils import binrvps_constantbins, binrvps, pairRegionsIntersection, rvps_to_rvps, binrvps_multi_interval
from .file_io import rvps_from_bed, rvps_from_npz, rvps_to_npz
import numpy as np
import pandas as pd
import pyBigWig as pBW
import math

class DataTrack():
    '''
    Base class for datatracks. If C is the set of chromosomes, Datatracks are functions:
                            
                            f: C x N ---> R
                            
    Which take in a (chromosome, integer) pair and return a value. How this function is
    encoded will depend on the input data.
        - Bed files will use region-value pairs and this will be implemented in a
          DataTrack_bed subclass. 
        - BigWig files will contain all reads overlapping with a given basepair and 
          accessing the data within them will be slightly different
          
    Every datatrack subclass should specify:
        - func: A function which takes in a chromosome name and a region and returns the
                function values for every basepair in that interval
        - stats: A function which takes in a chromosome name and a region and returns some
                 statistic of the function over that region e.g. mean or max
        - data_in_interval: Returns all the basic input data corresponding to an interval
                            on a given chromosome.
        - bin_dtrack_single_chr: Given some binSize and chromosome extent, bins a data
                                 track along a chromosome
        - bin_dtrack: Given some binSize, bins data along all chromosomes.
                 
                
    '''
    def __init__(self,
                 name,
                 dtrack_type = None,
                 tpoint = None,
                 chromosomes = None,
                 chr_lims = None
                ):
        self.name = name
        self.tpoint = tpoint
        self.dtrack_type = dtrack_type
        if chromosomes is None:
            self.chromosomes = []
        else:
            self.chromosomes = chromosomes
        if chr_lims is None:
            self.chr_lims = {}
        else:
            self.chr_lims = chr_lims
        
    def func(self,
             chr_name,
             region):
        return None
    
    def stats(self,
              chr_name,
              region,
              stats_type = 'mean'):
        return None
    
    def data_in_interval(self,
                         interval):
        return None
    
 
 
class DataTrack_rvp(DataTrack):
    def __init__(self,
                 name,
                 regions = None,
                 values = None,
                 IDs = None,
                 params = None,
                 **kwargs):
        '''
        Initialisation of a region-value-pair datatrack.
        Arguments:
        - Required:
            - Name: This is just the name of the track we are importing
        
        - Optional:
            - regions: (N,2) shape int32 array detailing the regions 
            - values: (N,1) shape double array detailing the values in each region
            - IDs: (N,1) shape array detailing the ID of each region
            - params: 
        
        '''
        super(DataTrack_rvp, self).__init__(name, dtrack_type = 'region-value-pairs', **kwargs)
        if regions is None:
            self.regions = {}
        else:
            self.regions = regions
        if values is None:
            self.values = {}
        else:
            self.values = values
        if IDs is None:
            self.IDs = {}
        else:
            self.IDs = IDs
            
        if params is None:    
            self.params = np.array([5]).astype(int)
        else:
            self.params = params
            
    def chrlims_from_regions(self):
        for chrom in self.regions:
            self.chr_lims[chrom] = [np.min(self.regions[chrom]), np.max(self.regions[chrom])]
        
    def func(self,
             chr_name, 
             mids,
             buffer = 1000,
             binSize = 1,
             use_constant_val = None,
             **kwargs):
        '''
        Evaluate our datatrack for all basepairs in a given region on a given chromosome
        Arguments:
            - chr_name: Name of the chromosome.
            - mids: The middles of the regions we are using for function evaluation
            - buffer: The buffer we want to add to each region on either side of the midpoint
            - binSize: The size of the bins to use when evaluating the function
            - use_constant_val: integer. If specified then datatrack values are replaced with
                                a constant.        
            - **kwargs: Extra argument passed to binrvps_multi_interval 

        '''
        mids = np.array(mids).astype('int32')
        if chr_name not in self.regions:
            return np.zeros((mids.shape[0],2*buffer))
        myvals = self.values[chr_name][:,0]
        if use_constant_val is not None:
            myvals = use_constant_val*np.ones(myvals.shape)
            
        return binrvps_multi_interval(self.regions[chr_name],
                                      myvals,
                                      mids,
                                      buffer,
                                      binSize,
                                      **kwargs)
    
    def stats(self,
              chr_name, 
              intervals,
              stats_type = 'mean',
              stats_scale = 1,
              void_regions = None,
              **kwargs):
        '''
        Evaluate our datatrack for all basepairs in a given region interval
        Arguments:
            - chr_name: Name of the chromosome.
            - interval: (N,2) shape array detailing the intervals we wan't to collect stats about.
            - stats_type: The type of statistic to gather over a particular region. This
                          is string valued and can be any one of:
                              - 'mean': The mean datatrack value over basepairs in the
                                        region.
                              - 'sum': The sume of datatrack values over basepairs in the
                                       region.
                              - 'coverage': The total number of basepairs with non-zero
                                            values in the reigon.
                              - 'max': The maximum datatrack value over basepairs in the
                                       region.
                              - 'min': The maximum datatrack value over basepairs in the
                                       region.
                              - 'std': The standard deviation of  datatrack value over 
                                       basepairs in the region.
                              - 'per_region_mean': The mean of the datatrack values
                                                   over regions within our intervals. 
                                                   If there are no regions within an
                                                   interval then it returns zero.
                              - 'per_region_min': The min of the datatrack values
                                                  over regions within our intervals. 
                                                  If there are no regions within an
                                                  interval then it returns zero.
                              - 'per_region_std': The standard deviation of the datatrack
                                                  values over regions within our intervals. 
                                                  If there are no regions within an
                                                  interval then it returns zero.
                          The first three options should be pretty quick to compute since
                          they don't require actually binning at basepair resolution. Users
                          should avoid using the latter three options for very large regions.
                          For the min and max functions if users really want to know the max
                          or min values over a large region then they can compute per-region
                          max and min values for much smaller regions and this could be quicker.
             - stats_scale: Since calculating statistics such as the per basepair max requires
                            calculating the binning for each basepair in every region, with very
                            large output regions this can be very memory intensive. If this is
                            an issue we can set the stats scale to be larger - e.g. calculating
                            the maximum value over each 100 basepair region.
             - kwargs: Other keyword arguments for regionBinValues
        '''
        
        if stats_type == 'mean':
            stats_type = 0
        elif stats_type == 'sum':
            stats_type = 1
        elif stats_type == 'min':
            stats_type = 2
        elif stats_type == 'max':
            stats_type = 3
        elif stats_type == 'std':
            stats_type = 4
        elif stats_type == 'coverage':
            stats_type = 5
        elif stats_type == 'per_region_mean':
            stats_type = 6
        elif stats_type == 'per_region_std':
            stats_type = 7
        elif stats_type == 'per_region_min':
            stats_type = 8
        elif stats_type in [0,1,2,3,4,5,6,7,8]:
            pass
        else:
            print("Unrecognised stats_type. Please pick from: 'mean','sum', 'max','min','std','coverage'")
            return None
        
        if chr_name not in self.regions:
            print("Cannot find {} in datatrack regions for {}. Returning zeros".format(chr_name, self.name))
            return np.zeros(intervals.shape[0])
        regions = self.regions[chr_name]
        values = self.values[chr_name][:,0]
        
        crc = np.array(regions).astype('int32')
        if void_regions is not None:
            idx = pairRegionsIntersection(crc,np.array(void_regions).reshape(-1,2).astype('int32'),
                                          exclude=True, allow_partial = False)
        else:
            idx = np.arange(len(crc))
            
        values = values[idx]
        regions = crc[idx]   
        stats = rvps_to_rvps(regions.astype('int32'),
                            values.astype('double'),
                            np.array(intervals).reshape(-1,2).astype('int32'),
                            stats_type = stats_type,
                            stats_scale = stats_scale)
        return stats
    
    def data_in_interval(self,chr_name, interval = None):
        '''
        Return raw data which overlaps with a given interval on a chromosome:
            - chr_name: Name of the chromosome.
            - interval: List or tuple of length 2 detailing the interval for which
                        we want to return data for. 
        '''
        out_dict = {}
        out_dict['regions'] = self.regions[chr_name]
        out_dict['values'] = self.values[chr_name]
        try:
            out_dict['IDs'] = self.IDs[chr_name]
        except:
            pass
        if interval is None:
            return out_dict
        else:   
            interval = np.array(interval).astype('int32').reshape((1,2))
        
            idxs = pairRegionsIntersection(self.regions[chr_name], interval, allow_partial = True)
        
            out_dict['regions'] = out_dict['regions'][idxs]
            out_dict['values'] = out_dict['values'][idxs]
            try:
                out_dict['IDs'] = out_dict['IDs'][idxs]
            except:
                pass
            return out_dict
    
    def bin_single_interval(self,
                       chr_name,
                       bins,
                       interval= None,
                       void_regions = None,
                       stats_type = 'mean',
                       stats_scale = 1,
                       **kwargs):
        '''
        Evaluate our datatrack for all basepairs in a given region on a given chromosome
        Arguments:
            - chr_name: Name of the chromosome.
            - bins: Can either be an integer specifigying binSize, an (N+1,) shape array detailing the bin starts where
                    N is our number of bins. In this case there are N+1 items since we must also specify the end of the
                    final bin. Otherwise users can put in an (N,2) shape array detailing the exact start and end of each
                    bin. Note that we still use the semi-open convention where a bin represents the semi open interval 
                    [a,b)
            - interval: If a single number is used as the bins input then that is used as a constant binSize for the binning
                        and this interval is the specify the maximum and minimum allowed values for the binning. All regions
                        [a,b] with b < interval[0] or a >= interval[1] are excluded from the binning. Regions with
                        a < extent[0] < b are clipped - the proportion of the bin overlapping with the allowed interval 
                        defined by extent is multiplied by the value of the original region.
            - stats_type: The type of statistic to gather over a particular region. This
                          is string valued and can be any one of:
                              - 'mean': The mean datatrack value over basepairs in the
                                        region.
                              - 'sum': The sume of datatrack values over basepairs in the
                                       region.
                              - 'coverage': The total number of basepairs with non-zero
                                            values in the reigon.
                              - 'max': The maximum datatrack value over basepairs in the
                                       region.
                              - 'min': The maximum datatrack value over basepairs in the
                                       region.
                              - 'std': The standard deviation of  datatrack value over 
                                       basepairs in the region.
                              - 'per_region_mean': The mean of the datatrack values
                                                   over regions within our intervals. 
                                                   If there are no regions within an
                                                   interval then it returns zero.
                              - 'per_region_min': The min of the datatrack values
                                                  over regions within our intervals. 
                                                  If there are no regions within an
                                                  interval then it returns zero.
                              - 'per_region_std': The standard deviation of the datatrack
                                                  values over regions within our intervals. 
                                                  If there are no regions within an
                                                  interval then it returns zero.
                          The first three options should be pretty quick to compute since
                          they don't require actually binning at basepair resolution. Users
                          should avoid using the latter three options for very large regions.
                          For the min and max functions if users really want to know the max
                          or min values over a large region then they can compute per-region
                          max and min values for much smaller regions and this could be quicker.
             - stats_scale: Since calculating statistics such as the per basepair max requires
                            calculating the binning for each basepair in every region, with very
                            large output regions this can be very memory intensive. If this is
                            an issue we can set the stats scale to be larger - e.g. calculating
                            the maximum value over each 100 basepair region. 
             - **kwargs: Extra arguments passed to rvps_to_rvps function.
              
        '''

        if type(bins) == int or type(bins) == float:
            if interval is None:
                if chr_name not in self.chr_lims:
                    print("No interval given. Inferring interval from regions...")
                    interval = np.array([np.min(self.regions[chr_name]), np.max(self.regions[chr_name])])
                else:
                    interval = np.array(self.chr_lims[chr_name])
                    
            newbins = np.append(np.arange(interval[0], interval[1], bins)[:,None],
                                np.arange(interval[0]+bins, interval[1]+bins, bins)[:,None], axis =1)
                
        elif len(bins.shape) == 1:
            newbins = np.append(bins[:-1,None],bins[1:,None], axis =1)
        else:
            newbins = bins
            
        return newbins, self.stats(chr_name,newbins,
                                   stats_type = stats_type,
                                   stats_scale = stats_scale,
                                   void_regions = void_regions,
                                   **kwargs)
    
    def bin_dtrack(self,
                   binSize,
                   chr_lims = None,
                   void_regions = None,
                   stats_type = 'mean',
                   stats_scale = 1,
                   **kwargs):
        '''
        Evaluate our datatrack for all basepairs in a given region on all our chromosomes.
        Arguments:
            - binSize: Integer. We use a constant binSize for the whole genome
            - chr_lims: Dictionary detailing specific chromosome limits for each chromosome
            - void_regions: Dictionary detailing specific regions to exclude on each chromosome
            - stats_type: The type of statistic to gather over a particular region. This
                          is string valued and can be any one of:
                              - 'mean': The mean datatrack value over basepairs in the
                                        region.
                              - 'sum': The sume of datatrack values over basepairs in the
                                       region.
                              - 'coverage': The total number of basepairs with non-zero
                                            values in the reigon.
                              - 'max': The maximum datatrack value over basepairs in the
                                       region.
                              - 'min': The maximum datatrack value over basepairs in the
                                       region.
                              - 'std': The standard deviation of  datatrack value over 
                                       basepairs in the region.
                              - 'per_region_mean': The mean of the datatrack values
                                                   over regions within our intervals. 
                                                   If there are no regions within an
                                                   interval then it returns zero.
                              - 'per_region_min': The min of the datatrack values
                                                  over regions within our intervals. 
                                                  If there are no regions within an
                                                  interval then it returns zero.
                              - 'per_region_std': The standard deviation of the datatrack
                                                  values over regions within our intervals. 
                                                  If there are no regions within an
                                                  interval then it returns zero.
                          The first three options should be pretty quick to compute since
                          they don't require actually binning at basepair resolution. Users
                          should avoid using the latter three options for very large regions.
                          For the min and max functions if users really want to know the max
                          or min values over a large region then they can compute per-region
                          max and min values for much smaller regions and this could be quicker.
             - stats_scale: Since calculating statistics such as the per basepair max requires
                            calculating the binning for each basepair in every region, with very
                            large output regions this can be very memory intensive. If this is
                            an issue we can set the stats scale to be larger - e.g. calculating
                            the maximum value over each 100 basepair region. 
             - **kwargs: Extra arguments passed to rvps_to_rvps function. 
                      
        '''
        out = {'bins': {},
               'vals': {}
              }
        if void_regions is None:
            void_regions = {}
            
        if chr_lims is None:
            chr_lims = {}
        
        for chrom in self.chromosomes:
            if chrom not in void_regions:
                void_input = None
            else:
                void_input = void_regions[chrom]
                
            if chrom not in chr_lims:
                chr_lim_input = None
            else:
                chr_lim_input = chr_lims[chrom]
            
            out['bins'][chrom], out['vals'][chrom] = self.bin_single_interval(chrom, 
                                                                              binSize,
                                                                              interval = chr_lim_input,
                                                                              void_regions = void_input,
                                                                              stats_type = stats_type,
                                                                              stats_scale = stats_scale)
            
        return out
                              
    def plot_in_region(self,
                       interval,
                       viewing_chrom,
                       ax,
                       binSize = 1e3,
                       stats_type = 'mean',
                       return_highest_val = True,
                       col = None,
                       **kwargs):
        '''
        Plot the datatrack in some region
        '''
        xpos,vals = self.bin_single_interval(viewing_chrom,
                                              binSize,
                                              interval = interval,
                                              stats_type = stats_type,
                                              **kwargs)
    
        xpos = np.mean(xpos,axis = 1)
    
        if col is None:
            ax.plot(xpos, np.zeros(len(xpos)), **kwargs)
            ax.fill_between(xpos, vals)
        else:
            ax.plot(xpos, np.zeros(len(xpos)), color = col, **kwargs)
            ax.fill_between(xpos,vals, color = col)
    
        if return_highest_val:
            return np.max(vals)
        else:
            return None
        
    def from_region_value_id_dicts(self,
                                   regs,
                                   vals,
                                   IDs = None):
        '''
        Generate region-value-pair track data dictionaries of region-value pairs
        Arguments:
            - regs: Dictionary with chromosome names as keys and a (N,2) shape array
                    for each chromosome where each row is a region and we have N
                    regions in that chromosome.
            - vals: Dictionary with chromosome names as keys and a (N,1) shape array
                    for each chromosome where the ith row is the value corresponding
                    and to the ith region for that chromosome.
            - IDs: Dictionary with chromosome names as keys and a (N,1) shape array
                    for each chromosome where the ith row is a unique ID corresponding
                    and to the ith region for that chromosome.
        '''
        if IDs is None:
            IDs = {}
        for chrom in regs:      
            if chrom not in self.chromosomes:
                self.regions[chrom] = regs[chrom].astype('int32')
                self.values[chrom] = vals[chrom].astype('double')
                if chrom in IDs:
                    self.IDs[chrom] = IDs[chrom]
                self.chromosomes.append(chrom)
                    
        for chrom in self.chromosomes:
            if chrom not in self.regions:
                self.regions[chrom] = np.empty((0,2)).astype('int32')
                self.values[chrom] = np.empty((0,1)).astype('double')
                if len(IDs.keys()) > 0:
                    self.IDs[chrom] = np.empty((0,1))
                    
        return self
        
    def from_npz(self,
                 npz_file,
                 params = False,
                 **kwargs):
        '''
        Generate region-value-pair track data (e.g. ChIp-seq) from Numpy archive (.npz)
        Arguments:
            - file_path: Path of the data track to be loaded
            - params: Boolean. If true then search the data archive for a 'params'
                      key and return that. The params dictionary is used to specify
                      any default parameters to be used when binning a datatrack. If
                      params is true then the params dictionary will be saved as a
                      datatrack attribute
            - **kwargs: Extra arguments to be passed to io.load_data_track
        '''
        if params:
            self.params = rvps_from_npz(npz_file, params = params).astype(int)
            params = False
        else:
            self.params = np.array([5]).astype('int')
        
        regs, vals, IDs = rvps_from_npz(npz_file, **kwargs)
        
        return self.from_region_value_id_dicts(regs,vals, IDs)
                    
                    
    def from_bed(self,
                 bed,
                 add_chroms = True,
                 **kwargs):
        '''
        Generate region-value-pair track data (e.g. ChIp-seq) from bed archive (.bed)
        Arguments:
            - file_path: Path of the data track to be loaded
            - **kwargs: Extra arguments to be passed to dtrack_io.load_data_track_from_bed
        '''
        regs, vals, IDs = rvps_from_bed(file_path = bed, **kwargs)
        
        return self.from_region_value_id_dicts(regs, vals, IDs)

    def to_npz(self, out_path, name = None):
        '''
        Output a datatrack to a .npz archive
        '''
        inputIDs = None
        if len(self.IDs.keys()) != 0:
            inputIDs = self.IDs
            
        inputparams = self.params
            
        if name is None:
            name = self.name
            
        rvps_to_npz(self.regions,
                    self.values,
                    name,
                    out_path,
                    IDs = inputIDs,
                    params = inputparams)
        
    def to_bed(out_path,
               name = None,
               sep = "\t"):
        '''
        Output a datatrack to a BED file
        '''
        inputIDs = None
        if len(self.IDs.keys()) != 0:
            inputIDs = self.IDs
        
        if name is None:
            name = self.name
        
        rvps_to_bed(self.regions,
                    self.values,
                    name,
                    out_path,
                    IDs = inputIDs,
                    sep = sep)
        
####################################################################################################
####################################################################################################
####################################################################################################
                    
class DataTrack_bigwig(DataTrack):                
    def __init__(self, name, **kwargs):
        super(DataTrack_bigwig, self).__init__(name, dtrack_type = 'BigWig', **kwargs)

    def from_bw(self, bw_file):
        self.data = pBW.open(bw_file)
        chrom_data = self.data.chroms()
        for key in chrom_data:
            self.chromosomes.append(key)
            self.chr_lims[key] = [0,chrom_data[key]]
        
        self.header = self.data.header()
        
        return self
        
    def func(self,
             chr_name, 
             mids,
             buffer = 1000,
             binSize = 1,
             **kwargs):
        '''
        Evaluate our datatrack for all basepairs in a given region on a given chromosome
        Arguments:
            - chr_name: Name of the chromosome.
            - mids: The middles of the regions we are using for function evaluation
            - buffer: The buffer we want to add to each region on either side of the midpoint
            - binSize: The size of the bins to use when evaluating the function       
            - **kwargs: Extra argument passed to the stats method

        '''
        mids = np.array(mids).astype('int32')
        out = []
        for mid in mids:
            out.append(self.data.stats(chr_name,
                                       mid-buffer,
                                       mid+buffer,
                                       nBins = int(2*buffer/binSize),
                                       **kwargs))
        
        return np.array(out)
                       
    def stats(self,
              chr_name, 
              intervals,
              fill_nan = 0,
              stats_type='mean',
              **kwargs):
        '''
        Evaluate our datatrack for all basepairs in a given region interval
        Arguments:
            - chr_name: Name of the chromosome.
            - Intervals: list of length two lists (or an (N,2) array) detailing
                         the intervals over which we want to collect statsitics.
             - kwargs: Other keyword arguments for the pyBigWig .stats method
        '''
         
        out = []
        for interval in intervals:
            num = self.data.stats(chr_name,
                                  int(interval[0]),
                                  int(interval[1]),
                                  type = stats_type,
                                  **kwargs)
            for idx, item in enumerate(num):
                if item is None:
                    num[idx] = fill_nan
                elif math.isnan(item):
                    num[idx] = fill_nan
            out.append(num)
                       
        return np.array(out)[:,0]
                       
    def data_in_interval(self,
                         chr_name, 
                         interval = None):
        '''
        Return raw data which overlaps with a given interval on a chromosome:
            - chr_name: Name of the chromosome.
            - interval: List or tuple of length 2 detailing the interval for which
                        we want to return data for. 
        '''
        if interval is None:
            return self.data.intervals(chr_name)
        else:
            return self.data.intervals(chr_name, interval[0], interval[1])
         
                       
    def bin_single_interval(self,
                            chr_name,
                            bins,
                            interval= None,
                            norm_signal_by_chrom_max = False,
                            fill_none = 0,
                            stats_type = 'mean',
                            **kwargs):
        '''
        Evaluate our datatrack for all basepairs in a given region on a given chromosome
        Arguments:
            - chr_name: Name of the chromosome.
            - bins: Can either be an integer specifigying binSize, an (N+1,) shape array detailing the bin starts where
                    N is our number of bins. In this case there are N+1 items since we must also specify the end of the
                    final bin. Otherwise users can put in an (N,2) shape array detailing the exact start and end of each
                    bin. Note that we still use the semi-open convention where a bin represents the semi open interval 
                    [a,b)
            - interval: If a single number is used as the bins input then that is used as a constant binSize for the binning
                        and this interval is the specify the maximum and minimum allowed values for the binning. All regions
                        [a,b] with b < interval[0] or a >= interval[1] are excluded from the binning. Regions with
                        a < extent[0] < b are clipped - the proportion of the bin overlapping with the allowed interval 
                        defined by extent is multiplied by the value of the original region.
             - **kwargs: Extra arguments passed to the stats function.
              
        '''
        
        if chr_name not in self.chr_lims and 'chr' not in chr_name:
            chr_name = "chr" + chr_name
        
        if chr_name not in self.chr_lims:
            raise Exception('Given chromosome not in this dataset')
            
        if type(bins) == int:
            if interval is None:
                if chr_name not in self.chr_lims:
                    print("No interval given. Inferring interval from regions...")
                    return None
                else:
                    interval = np.array(self.chr_lims[chr_name])
                    
            newbins = np.append(np.arange(interval[0], interval[1], bins)[:,None],
                                np.arange(interval[0]+bins, interval[1]+bins, bins)[:,None],
                                axis =1)
            
            e = int((interval[1]-interval[0]-1)/bins)
            #Number of bins Must be at least one if e-s < binSize
            nBins = e+1
            
            out = self.data.stats(chr_name,
                                  int(interval[0]),
                                  int(interval[1]),
                                  nBins = nBins,
                                  type = stats_type,
                                  **kwargs)
            
            out = [item if item is not None else fill_none for item in out]
            out = np.array(out)
            if norm_signal_by_chrom_max:
                out /= self.data.stats(chr_name, type = 'max')
            return newbins, out
                
        elif len(bins.shape) == 1:
            newbins = np.append(bins[:-1,None],bins[1:,None], axis =1)
        else:
            newbins = bins
        
        out = np.full(newbins.shape,fill_none)
        for idx,mybin in enumerate(newbins):
            stat = self.data.stats(chr_name, 
                                   int(mybin[0]),
                                   int(mybin[1]),
                                   type = stats_type,
                                   **kwargs)
            if stat is not None:
                out[idx] = stat
        
        out = [item if item is not None else fill_none for item in out]
        out = np.array(out)
        if norm_signal_by_chrom_max:
            out /= self.data.stats(chr_name, type = 'max')
        
        return newbins, np.array(out)
                       
                       
    def bin_dtrack(self,
                   binSize,
                   chr_lims = None,
                   void_regions = None,
                   stats_type = 'mean',
                   stats_scale = 1,
                   **kwargs):
        '''
        Evaluate our datatrack for all basepairs in a given region on all our chromosomes.
        Arguments:
            - binSize: Integer. We use a constant binSize for the whole genome
            - chr_lims: Dictionary detailing specific chromosome limits for each chromosome
             - **kwargs: Extra arguments passed to pyBigWig stats function. 
                      
        '''
        out = {'bins': {},
               'vals': {}
              }
        if void_regions is None:
            void_regions = {}
            
        if chr_lims is None:
            chr_lims = {}
            
        for chrom in self.chromosomes:
            if chrom not in void_regions:
                void_input = None
            else:
                void_input = void_regions[chrom]
                
            if chrom not in chr_lims:
                chr_lim_input = None
            else:
                chr_lim_inputer = chr_lims[chrom]
            out['bins'][chrom], out['vals'][chrom] = self.bin_single_interval(chrom, binSize,
                                                                              interval= chr_lim_input,
                                                                              void_regions = void_input, 
                                                                              **kwargs)
            
        return out
    
    
    def plot_in_region(self,
                        interval,
                        viewing_chrom,
                        ax,
                        binSize = 1e3,
                        stats_type = 'mean',
                        norm_signal_by_chrom_max = True,
                        return_highest_val = True,
                        col = None,
                        **kwargs):
        '''
        Plot the datatrack in some region
        '''
        if viewing_chrom not in self.chromosomes:
            for chrom in self.chromosomes:
                if viewing_chrom == chrom[3:]:
                    viewing_chrom = chrom
                    break
        
        if viewing_chrom not in self.chromosomes:
            print("Couldn't find that chromosome in this file...")
            return None
        
        xpos,vals = self.bin_single_interval(viewing_chrom,
                                             binSize,
                                             interval = interval,
                                             norm_signal_by_chrom_max = norm_signal_by_chrom_max,
                                             stats_type = stats_type,
                                                )
    
        xpos = np.mean(xpos,axis = 1)
    
        if col is None:
            ax.plot(xpos, np.zeros(len(xpos)), **kwargs)
            ax.fill_between(xpos, vals)
        else:
            ax.plot(xpos, np.zeros(len(xpos)), color = col, **kwargs)
            ax.fill_between(xpos,vals, color = col)
    
        if return_highest_val:
            return np.max(vals)
        else:
            return None