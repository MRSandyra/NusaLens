import sys
import pyarrow.parquet as pq

f = pq.ParquetFile('data/gold/chunks.parquet')
print('chunks rows', f.metadata.num_rows)
