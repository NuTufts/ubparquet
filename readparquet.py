import os,sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

class UBParquetReader:
    def __init__(self,filename):
        self.table = pq.read_table(filename)
        print("columns: ",self.table.column_names)
        print("nrows: ",self.table.num_rows)
        self.nentries = self.table.num_rows


    def get_entry(self,ientry):
        if ientry<0 or ientry>=self.nentries:
            print("Entry out of range. Table contains %d entries."%(self.nentries))
            return {}

        #print("ENTRY: ",ientry)
        entry_data = {}
        for k in self.table.column_names:
            if "_shape" in k:
                continue
            #print("unpack: ",k)
            shape_name = k+"_shape"
            if shape_name in self.table.column_names:
                s = self.table[shape_name][ientry].as_py()
                shape = np.array( s, dtype=np.int )
                d = self.table[k][ientry].as_py()
                arr = np.array( d ).reshape( shape )
                entry_data[k] = arr
            else:
                entry_data[k] = self.table[k][ientry]
        return entry_data


if __name__ == "__main__":

    reader = UBParquetReader( "temp.parquet" )
    entry_data = reader.get_entry(0)    
    print(entry_data['matchtriplet'][:10,:])
    print(entry_data['spacepoint_t'][:10,:])    
