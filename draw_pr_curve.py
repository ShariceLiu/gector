from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

def computeFScore(tp, fp, fn, beta = 1.0):
    if tp:
        p = float(tp)/(tp+fp) if fp else 1.0
        r = float(tp)/(tp+fn) if fn else 1.0
        f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
    else:
        p = 0.0 if fp else 1.0
        r = 0.0 if fn else 1.0
        f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)

def draw_pr_curve(filename,col = 0):
    '''
    draw pr curve according to file 
    with first column to be the probability/confidence, second col to be the true labels
    '''
    with open(filename,'r',encoding='UTF-8') as f:
        lines = f.readlines()

    probs = [float(l.rstrip('\n').split()[col]) for l in lines]
    # print(probs)
    probs = np.array(probs)
    probs[probs==float('inf')] = 1e9
    labels = [int(l.rstrip('\n').split()[1]) for l in lines]

    precisions, recalls, _ = precision_recall_curve(labels, probs)

    prec_rec = list(zip(precisions,recalls))
    prec_rec = list(filter(lambda x: (x[0] != 0.0 and x[0] != 1.0 and x[1] != 0.0 and x[1] != 1.0),prec_rec))
    precisions = np.array(list(zip(*prec_rec))[0])
    recalls = np.array(list(zip(*prec_rec))[1])

    return precisions, recalls

if __name__ == '__main__':
    filename = "/home/zl437/rds/hpc-work/gector/fce-data/with_prob/text_pred_class.txt"
    filename_2 = '/home/mifs/zl437/gector/data/test_pred/text_pred_conll14_d_tag.txt'
    filename_nucle_baseline = '/home/zl437/rds/hpc-work/gector/bea-data/nucle/with_prob/test_pred_bea.txt'
    filename_fce_baseline = '/home/zl437/rds/hpc-work/gector/fce-data/with_prob/test_pred_class_fce.txt'
    # precisions, recalls = draw_pr_curve(filename)
    # plt.plot(recalls*100,precisions*100)
    precisions, recalls = draw_pr_curve(filename_nucle_baseline)
    plt.plot(recalls*100,precisions*100)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("PR curve")
    # plt.legend(['class prob', 'tag prob'])

    plt.show()
    plt.savefig("/home/zl437/rds/hpc-work/gector/bea-data/nucle/with_prob/PR_curve_nucle.png")
    print("saved")