from numpy import array,where,sum,isclose

def precision_recall(predictions,ground_truths,atol=0.07) :

    # construct overlap matrix
    tp_matrix = array([[ isclose(p,gt,atol=atol) for gt in ground_truths ] for p in predictions ])

    # pair up detections and ground truths
    i = 0
    for tp in tp_matrix.T :
        tp_index, = where(tp)

        for n,j in enumerate(tp_index) :
            if n is 0 :
                tp_matrix[j] = array(len(tp_matrix.T)*[False])
                tp_matrix[j,i] = True
            else :
                tp_matrix[j,i] = False

        i += 1

    recall = sum(tp_matrix)/len(ground_truths)
    precis = sum(tp_matrix)/len(predictions)
    return tp_matrix,precis,recall
