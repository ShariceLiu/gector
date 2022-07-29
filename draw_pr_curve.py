from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

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
    filename = "data/test_pred/test_pred.txt"
    precisions, recalls = draw_pr_curve(filename)

    plt.plot(recalls*100,precisions*100)
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("PR curve")

    plt.show()
    plt.savefig("PR_curve.png")