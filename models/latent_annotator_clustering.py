import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

# to simplify, all annotators will annotate each data point once,
# but the data structure allows for anything else
def data_generator(num_classes, num_annotators, num_datapoints, num_clusters):
    # annotator cluter
    alpha = np.ones(num_clusters) 
    gold_B = np.random.dirichlet(alpha, size=num_annotators)

    # gold_B = np.zeros((num_annotators, num_clusters))
    # for a in range(num_annotators):
    #     gold_cluster = np.random.randint(0, num_clusters)
    #     gold_B[a, gold_cluster] = 1

    # annotation distribution for each cluster and class
    beta = np.ones(num_classes) 
    gold_C = np.random.dirichlet(beta, size=(num_clusters, num_classes))
    
    # gold_C = np.zeros((num_clusters, num_classes, num_classes))
    # for g in range(num_clusters):
    #     for k in range(num_classes):
    #         gold_class = np.random.randint(0, num_classes)
    #         gold_C[g, k, gold_class] = 1
    
    # data point class
    gold_A = np.zeros((num_datapoints, num_classes))
    data = list()
    for i in range(num_datapoints):
        gold_class = np.random.randint(0, num_classes)
        gold_A[i][gold_class] = 1
        data.append(list())
        for a in range(num_annotators):
            #cluster g
            g = np.random.multinomial(1, gold_B[a]).argmax()
            data[-1].append((a, np.random.multinomial(1, gold_C[g, gold_class]).argmax()))
            
    return data, gold_A, gold_B, gold_C

def EM(data, num_classes, num_annotators, num_clusters, num_iters=20, smoothing=True):
    num_datapoints = len(data)
    
    # Random initialization
    # Warning: each row must sum to 1 in the two amtrices
    A = np.random.rand(num_datapoints, num_classes)
    A /= A.sum(axis=1, keepdims=True)

    B = np.random.rand(num_annotators, num_clusters)
    B /= B.sum(axis=1, keepdims=True)

    C = np.random.rand(num_clusters, num_classes, num_classes)
    C /= C.sum(axis=2, keepdims=True)

    # Approximate posterior q
    # we need one vector per annotation
    # no need to initialise because we start with the E step
    q = [[np.empty((num_classes, num_clusters)) for _ in annotations] for annotations in data]

    smoothing_value = 0.1/num_classes

    elbos = []
    for _ in range(num_iters):
        # E step
        # update distribution q
        e = 0
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            for (a, c), item_q in zip(item_annotations, item_qs):
                # WARNING: inplace update
                for cc, row in enumerate(item_q):
                    for g, item in enumerate(row):
                        item_q[cc, g] = A[item_idx, cc] * B[a, g] * C[g, cc, c]
                t = np.log(item_q + 1e-10)
                item_q /= item_q.sum()
                entropy = - (item_q * np.log(item_q + 1e-10)).sum()
                t1 = (item_q * t).sum()
                e += t1 + entropy
        elbos.append(e)


        # M step
        # update model parameters
        B.fill(0.)
        C.fill(0.)
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            # start with A, we update one item at a time
            A[item_idx] = np.sum(item_qs, axis = (0,2))
            if(smoothing):
                A[item_idx] = (A[item_idx] + smoothing_value) / (A[item_idx] + smoothing_value).sum()
            else:
                A[item_idx] /= A[item_idx].sum()

            for (a, c), item_q in zip(item_annotations, item_qs):
                B[a, :] += np.sum(item_q, axis = 0)
                C[:, :, c] += item_q.T
        if(smoothing):
            B = (B+smoothing_value) / (B+smoothing_value).sum(axis=1, keepdims=True)
            C = (C+smoothing_value) / (C+smoothing_value).sum(axis=2, keepdims=True)
            # T = (eta_1 / eta_2 +smoothing_value) / (eta_1 / eta_2 + 1 + smoothing_value)
        else:
            B /= B.sum(axis=1, keepdims=True)
            C /= C.sum(axis=2, keepdims=True)
        # B renormalization
        # B /= B.sum(axis=1, keepdims=True)
        # C renormalization
        # C /= C.sum(axis=2, keepdims=True)

    return A, B, C, elbos

def EM_logspace(data, num_classes, num_annotators, num_clusters, num_iters=20, smoothing=True):
    num_datapoints = len(data)
    
    # Random initialization
    # Warning: each row must sum to 1 in the two amtrices
    A = np.random.rand(num_datapoints, num_classes)
    A /= A.sum(axis=1, keepdims=True)
    A = np.log(A)

    B = np.random.rand(num_annotators, num_clusters)
    B /= B.sum(axis=1, keepdims=True)
    B = np.log(B)

    C = np.random.rand(num_clusters, num_classes, num_classes)
    C /= C.sum(axis=2, keepdims=True)
    C = np.log(C)

    # Approximate posterior q
    # we need one vector per annotation
    # no need to initialise because we start with the E step
    q = [[np.empty((num_classes, num_clusters)) for _ in annotations] for annotations in data]

    smoothing_value = 0.1/num_classes

    elbos = []
    for _ in range(num_iters):
        # E step
        # update distribution q
        e = 0
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            for (a, c), item_q in zip(item_annotations, item_qs):
                # WARNING: inplace update
                for cc, row in enumerate(item_q):
                    for g, item in enumerate(row):
                        item_q[cc, g] = A[item_idx, cc] + B[a, g] + C[g, cc, c]
                t = item_q.copy()
                item_q -= logsumexp(item_q)
                entropy = - (np.exp(item_q) * item_q).sum()
                t1 = (np.exp(item_q) * t).sum()
                e += t1 + entropy
        elbos.append(e)


        # M step
        # update model parameters
        B.fill(float("-inf"))
        C.fill(float("-inf"))
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            # start with A, we update one item at a time
            # A[item_idx] = np.sum(item_qs, axis = (0,2))
            A[item_idx] = np.logaddexp.reduce(item_qs, axis = (0,2))
            if(smoothing):
                A[item_idx] = np.logaddexp(A[item_idx], np.log(smoothing_value)) - logsumexp(np.logaddexp(A[item_idx], np.log(smoothing_value)))
            else:
                A[item_idx] -= logsumexp(A[item_idx])


            for (a, c), item_q in zip(item_annotations, item_qs):
            
                B[a, :] = np.logaddexp(B[a, :], np.logaddexp.reduce(item_q, axis = 0))

                C[:, :, c] = np.logaddexp(C[:, :, c], item_q.T)

        if(smoothing):
            B = np.logaddexp(B, np.log(smoothing_value)) - logsumexp(np.logaddexp(B, np.log(smoothing_value)), axis=1, keepdims=True)
            C = np.logaddexp(C, np.log(smoothing_value)) - logsumexp(np.logaddexp(C, np.log(smoothing_value)), axis=2, keepdims=True)
  
        else:
            B -= logsumexp(B, axis=1, keepdims=True)
            C -= logsumexp(C, axis=2, keepdims=True)


    return np.exp(A), np.exp(B), np.exp(C), elbos

def evaluate(data, num_classes, num_annotators, num_clusters, num_iters=20, num_restarts=10, logspace=True, smoothing=True):

    best_elbo = -float("inf")
    best_A = None
    best_B = None
    best_C = None
    best_elbos = None
    for i in range(num_restarts):
        if logspace:
            A, B, C, elbos = EM_logspace(data, num_classes, num_annotators, num_clusters, num_iters, smoothing)
        else:
            A, B, C, elbos = EM(data, num_classes, num_annotators, num_clusters, num_iters, smoothing)
        if elbos[-1] > best_elbo:
            best_elbo = elbos[-1]
            best_A = A
            best_B = B
            best_C = C
            best_elbos = elbos
    # print("Best Accuracy: ", accuracy)
    return best_A, best_B, best_C, best_elbos

