import os,time
import copy
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class larmatchDataset(torch.utils.data.Dataset):
    def __init__(self, source=None, verbose=False, load_all_columns=True ):
        """
        Parameters:

        EXPECTED SCHEMA OF UBPARQUTE FILES
 |-- matchtriplet: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- matchtriplet_shape: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- ssnet_label: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- ssnet_label_shape: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- kplabel: array (nullable = true)
 |    |-- element: float (containsNull = true)
 |-- kplabel_shape: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- spacepoint_t: array (nullable = true)
 |    |-- element: float (containsNull = true)
 |-- spacepoint_t_shape: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- wireimg_feat0: array (nullable = true)
 |    |-- element: float (containsNull = true)
 |-- wireimg_feat0_shape: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- wireimg_coord0: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- wireimg_coord0_shape: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- wireimg_feat1: array (nullable = true)
 |    |-- element: float (containsNull = true)
 |-- wireimg_feat1_shape: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- wireimg_coord1: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- wireimg_coord1_shape: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- wireimg_feat2: array (nullable = true)
 |    |-- element: float (containsNull = true)
 |-- wireimg_feat2_shape: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- wireimg_coord2: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- wireimg_coord2_shape: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- run: long (nullable = true)
 |-- subrun: long (nullable = true)
 |-- event: long (nullable = true)
        """

        self._cols = ['wireimg_coord0',
                      'wireimg_coord0_shape',
                      'wireimg_feat0',
                      'wireimg_feat0_shape',
                      'wireimg_coord1',
                      'wireimg_coord1_shape',
                      'wireimg_feat1',
                      'wireimg_feat1_shape',
                      'wireimg_coord2',
                      'wireimg_coord2_shape',
                      'wireimg_feat2',
                      'wireimg_feat2_shape',
                      'matchtriplet',
                      'matchtriplet_shape',
                      'ssnet_label',
                      'ssnet_label_shape',
                      'kplabel',
                      'kplabel_shape',
                      'run','subrun','event']

        self.spark = SparkSession \
            .builder \
            .master('local') \
            .config("spark.driver.memory", "15g") \
            .appName("larmatchDataset") \
            .config('spark.executor.memory', '15gb') \
            .config("spark.cores.max", "6") \
            .getOrCreate()        
        
        print("larmatchDataset: loading dataset...")
        self._load_all_cols = load_all_columns
        self.df = self.spark.read.option("mergeSchema", "true").parquet(source)

        if not self._load_all_cols:
            self.df = self.df.select(*self._cols)
        else:
            self.df = self.df.select("*")

        print(" ... finished reading table ... table schema:")
        self.df.printSchema()
        self._nsample_per_attempt = 10
        
        # Load the parquet table
        #print("columns: ",self.df.column_names)
        #print("nrows: ",self.df.num_rows)
        self.nentries = self.df.count()
        print(" ... number of entries: ",self.nentries)

        self._verbose = verbose


        #self.random_access = random_access
        # self.partition_index = 0
        # self.num_partitions = 1
        # self.start_index = 0
        # self.end_index = self.nentries

        # self._current_entry  = self.start_index
        # self._nloaded        = 0
        # self._verbose = False
        # self._random_access = random_access
        # self._random_entry_list = None
        # if self._random_access:
        #     self._rng = np.random.default_rng(None)
        #     self._random_entry_list = self._rng.choice( self.nentries, size=self.nentries )

        # self._voxelize = voxelize
        # self._voxelsize_cm = voxelsize_cm
        # if self._voxelize and not self.is_voxeldata:
        #     self.voxelizer = larflow.voxelizer.VoxelizeTriplets()
        #     self.voxelizer.set_voxel_size_cm( self._voxelsize_cm )

    def dump_rse(self):
        self.df.select("run","subrun","event").show()

    def get_entry(self,run,subrun,event):
        sampled_rows = self.df.filter( "run==%d and subrun==%d and event==%d"%(run,subrun,event) ).rdd.collect()
        if len(sampled_rows)>0:
            return self.get_data_dict_from_ubparquet_file( sampled_rows )
        else:
            return {}

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()

        # get the entry from the parquet file
        s = time.time()
        count = 0
        while (count<1):
            sampled_rows = self.df.sample(False, float(self._nsample_per_attempt)/float(self.nentries)).limit(1)
            count = sampled_rows.count()
            print("counts returned: ",count)
            if count<1:
                print("WTF")
                
        dtsample = time.time()-s
        if self._verbose:            
            print("time to sample: ",dtsample)

        #sample = sampled_rows.select(*self._cols).collect()
        sample = sampled_rows.rdd.collect()
        data = self.get_data_dict_from_ubparquet_file( sample )
    
        # xlist = np.unique( data["voxcoord"], axis=0, return_counts=True )
        # indexlist = xlist[0]
        # counts = xlist[-1]
        # hasdupe = False
        # for i in range(indexlist.shape[0]):
        #     if counts[i]>1:
        #         print(i," ",indexlist[i,:]," counts=",counts[i])
        #         hasdupe = True
        # if hasdupe:
        #     raise("[larvoxel_dataset::__getitem__] Dupe introduced somehow in batch-index=%d"%(ibatch)," arr=",data["voxcoord"].shape)

        # self._nloaded += 1
        # self._current_entry += 1            
        
        # return copy.deepcopy(data)
        return data        

    def get_data_dict_from_ubparquet_file(self, df):
        """
        retrieve data from 
        """
        #print(df)

        # SLOW ASF
        #tpdf = time.time()
        #pdf = df.toPandas()
        #print(pdf)
        #print("time to pandas: ",time.time()-tpdf)
        
        tstart = time.time()

        #dfcol = df.select(*self._cols).collect()[0]
        dfcol = df[0]

        data = {"run":    dfcol['run'],
                "subrun": dfcol['subrun'],
                "event":  dfcol['event'] }

        for col in ["wireimg_coord0","wireimg_coord1","wireimg_coord2","wireimg_feat0","wireimg_feat1","wireimg_feat2"]:
            shape = np.array(dfcol[col+"_shape"])
            data[col] = np.array( dfcol[col] ).reshape(shape)

        if self._verbose:
             tottime = time.time()-tstart
             print("[larmatchDataset::get_data_dict_from_ubparquet_file]")
             print("  time from pyspark dataframe to numpy arrays: %.3f secs"%(tottime))
            
        return data
    
    def __len__(self):
        return self.nentries

    def print_status(self):
        print("worker: entry=%d nloaded=%d"%(self._current_entry,self._nloaded))

    def set_partition(self,partition_index,num_partitions):
        self.partition_index = partition_index
        self.num_partitions = num_partitions
        self.start_index = int(self.partition_index*self.nentries)/int(self.num_partitions)
        self.end_index   = int((self.partition_index+1)*self.nentries)/int(self.num_partitions)
        self._current_entry = self.start_index

    def collate_fn(batch):
        """
        """
        npoints = 0
        batch_size = len(batch)
        batch_dict = {}

        return batch_dict
    
            
if __name__ == "__main__":

    import time,sys
    import MinkowskiEngine as ME

    niter = 10
    batch_size = 1
    #test = larmatchDataset( filefolder="./temp.parquet", random_access=True )
    #test = larmatchDataset( source="./temp.parquet", random_access=True )
    test = larmatchDataset( source="./data/*/*.parquet", verbose=True, load_all_columns=False )
    nrows = len(test)
    print("num rows: ",nrows)

    loader = torch.utils.data.DataLoader(test,batch_size=batch_size)
    
    #test.dump_rse()

    start = time.time()
    for iiter in range(niter):

        print("====================================")
        #for ib,data in enumerate(batch):
        print("ITER[%d]"%(iiter))
        batch = next(iter(loader))
        #print("batch ncounts: ",batch.count())

    
    #     print("batch keys: ",batch.keys())
    #     print(batch["tree_entry"])
        
    #     for name,d in batch.items():
    #         if type(d) is np.ndarray:
    #             print("  ",name,": ",d.shape)
    #         else:
    #             print("  ",name,": ",type(d))

    #     print("iteration batch check ------------")
    #     xlist = np.unique( batch["voxcoord"], axis=0, return_counts=True )
    #     indexlist = xlist[0]
    #     counts = xlist[-1]
    #     hasdupe = False
    #     for i in range(indexlist.shape[0]):
    #         if counts[i]>1:
    #             print(i," ",indexlist[i,:]," ",counts[i])
    #             hasdupe = True
    #     if hasdupe:
    #         raise("iteration has dupe! -------------")

    #     coords = batch["voxcoord"]
    #     feats  = batch["voxfeat"]        
    #     A = ME.SparseTensor(features=torch.from_numpy(feats), coordinates=torch.from_numpy(coords))
    #     print("sparsetensor: ",A)            
        
    #     #xinput = batch["sparsetensor"]
    #     #print("after sparse collate")
    #     #print(xinput)
    #     print("====================================")
                                            
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)
    print("sec per iter: ",sec_per_iter)

    print("[enter] to finished.")
    input()
