from utils.helpers import read_parallel_lines

sourcepath = '/home/zl437/rds/hpc-work/gector/fce-data/train/source-train.txt'
targetpath = '/home/zl437/rds/hpc-work/gector/fce-data/train/target-train.txt'

sourcedata,targetdata = read_parallel_lines(sourcepath, targetpath)

with open(sourcepath, 'w') as g:
    g.write("\n".join(sourcedata) + '\n')
    
with open(targetpath, 'w') as g:
    g.write("\n".join(targetdata) + '\n')