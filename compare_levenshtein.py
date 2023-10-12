from Levenshtein import distance
from utils.helpers import read_lines

def get_levenshtein_d(file1, file2):
    source_data = read_lines(file1)
    target_data = read_lines(file2)
    
    assert len(source_data)==len(target_data)
    
    dist = 0
    for src, tgt in zip(source_data, target_data):
        dist += distance(src, tgt)
    
    return dist

if __name__ == '__main__':
    # file1 = '/home/zl437/rds/hpc-work/gector/fce-data/source-test.txt'
    file1 = '/home/zl437/rds/hpc-work/gector/fce-data/predict-train-3.txt'
    file2 = '/home/zl437/rds/hpc-work/gector/fce-data/predict-train-3-2nd.txt'
    d = get_levenshtein_d(file1, file2)
    print(d)