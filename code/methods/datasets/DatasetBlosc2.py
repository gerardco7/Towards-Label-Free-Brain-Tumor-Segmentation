import blosc2
import numpy as np
import os
import math
from copy import deepcopy

class DatasetBlosc2():
    '''
    nnUnet code for working with blosc2 files
    https://github.com/MIC-DKFZ/nnUNet
    '''
    def __init__(self, 
                 folder,
                 identifiers, 
                 #folder_with_segs_from_previous_stage
                 ):
        super(DatasetBlosc2, self).__init__()
        blosc2.set_nthreads(16)  
        
        if identifiers is None:
            identifiers = self.get_identifiers(folder)
        identifiers.sort()

        self.folder = folder
        self.identifiers = identifiers

    def __getitem__(self, identifier):
        return self.load_case(identifier)
    
    def load_case(self, identifier):
        dparams ={
            'nthreads': 16,
        }

        data_b2nd_file = os.path.join(self.folder, identifier + ".b2nd")
        #data = blosc2.open(urlpath=data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')
        data = blosc2.open(urlpath=data_b2nd_file, mode='r', dparams=dparams)

        seg_b2nd_file = os.path.join(self.folder, identifier + "_seg.b2nd")
        if os.path.exists(seg_b2nd_file):
            #seg = blosc2.open(urlpath=seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')
            seg = blosc2.open(urlpath=seg_b2nd_file, mode='r', dparams=dparams)
        else:
            seg = None

        return data, seg
    
    # staticmethod: does not require a class instance to be called
    @staticmethod
    def save_case(
        data, 
        seg,
        output_filename_truncated,
        chunks=None,
        blocks=None,
        chunks_seg=None,
        blocks_seg=None,
        clevel=8,
        codec=blosc2.Codec.ZSTD
    ):
        
        blosc2.set_nthreads(16)

        cparams = {
            'codec': codec,
            'clevel': clevel,
        }
        
        blosc2.asarray(np.ascontiguousarray(data), urlpath=output_filename_truncated + '.b2nd', chunks=chunks,
                       blocks=blocks, cparams=cparams, mmap_mode='w+')
        
        # For trainning data we will not save the seg
        if seg is not None:
            blosc2.asarray(np.ascontiguousarray(seg), urlpath=output_filename_truncated + '_seg.b2nd', chunks=chunks_seg,
                        blocks=blocks_seg, cparams=cparams, mmap_mode='w+')

    @staticmethod
    def get_identifiers(folder):
        """
        returns all identifiers in the preprocessed data folder
        """
        case_identifiers = [i[:-5] for i in os.listdir(folder) if i.endswith('.b2nd') and not i.endswith('_seg.b2nd')]
        return case_identifiers

    @staticmethod
    def comp_blosc2_params(
        image_size,
        patch_size,
        bytes_per_pixel = 4, # 4 bytes per pixel for float32# 
        # ***i9-9900K CPU has 8 cores and 16 threads***
        l1_cache_size_per_core_in_bytes = 64 * 1024, # 64 KB per core
        l3_cache_size_per_core_in_bytes = 2 * 1024 * 1024, # 2 MB per core
        safety_factor = 0.8 # 80% of cache size is used
    ):
        """
        Computes a recommended block and chunk size for saving arrays with blosc v2.

       
        Note: this is optimized for nnU-Net dataloading where each read operation is done by one core. We cannot use threading

        Cache default values computed based on old Intel 4110 CPU with 32K L1, 128K L2 and 1408K L3 cache per core.
        We cannot optimize further for more modern CPUs with more cache as the data will need be be read by the
        old ones as well.

        Args:
            ***image_size***: Image size, must be 4D (c, x, y, z). For 2D images, make x=1
            patch_size: Patch size, spatial dimensions only. So (x, y) or (x, y, z)
            bytes_per_pixel: Number of bytes per element. Example: float32 -> 4 bytes
            l1_cache_size_per_core_in_bytes: The size of the L1 cache per core in Bytes.
            l3_cache_size_per_core_in_bytes: The size of the L3 cache exclusively accessible by each core. 

        Returns:
            The recommended block and the chunk size.
        """

        num_channels = image_size[0]
        patch_size = np.array(patch_size)
        # ***Compute the minimum power of 2 that is greater than the patch size***
        block_size = np.array((num_channels, *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size]))

        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
            # pick largest deviation from patch_size that is not 1
            axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
            idx = 0
            picked_axis = axis_order[idx]
            # *** Same condition appears twice in the same loop ***
            # *** Picked axis + 1, because the first dimension is the number of channels ***
            while block_size[picked_axis + 1] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            # now reduce that axis to the next lowest power of 2
            block_size[picked_axis + 1] = 2 ** (max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1))))
            block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

        # ***In case the block size is larger than the image size, we need to adjust it***
        block_size = np.array([min(i, j) for i, j in zip(image_size, block_size)])

        # note: there is no use extending the chunk size to 3d when we have a 2d patch size! This would unnecessarily
        # load data into L3
        # now tile the blocks into chunks until we hit image_size or the l3 cache per core limit
        chunk_size = deepcopy(block_size)
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
            # ***If the patch size is 1, and y, z from the image_size is the same as the chunk_size***
            if patch_size[0] == 1 and all([i == j for i, j in zip(chunk_size[2:], image_size[2:])]):
                break
            # *** If the image_size is the same as the chunk_size***
            if all([i == j for i, j in zip(chunk_size, image_size)]):
                break
            # find axis that deviates from block_size the most
            axis_order = np.argsort(chunk_size[1:] / block_size[1:])
            idx = 0
            picked_axis = axis_order[idx]
            while chunk_size[picked_axis + 1] == image_size[picked_axis + 1] or patch_size[picked_axis] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
            chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel

            if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size)]) > 1.5:
                # chunk size should not exceed patch size * 1.5 on average
                chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
                break
        # better safe than sorry
        # ***In case the chunk size is larger than the image size, we need to adjust it***
        chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size)]

        return tuple(block_size), tuple(chunk_size)
    
