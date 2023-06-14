import numpy as np
from scipy.special import logsumexp
from .utils import softplus_np

# to simplify, all annotators will annotate each data point once,
# but the data structure allows for anything else
def data_generator(num_classes, num_annotators, num_datapoints):
    # annotator trustworthiness
    gold_T = np.random.beta(0.5,0.5, size=num_annotators)

    alpha = np.ones(num_classes)#*10
    # annotator spamming behaviour
    gold_B = np.random.dirichlet(alpha, size=num_annotators)

    # data point class
    gold_A = np.zeros((num_datapoints, num_classes))
    data = list()
    for i in range(num_datapoints):
        gold_class = np.random.randint(0, num_classes)
        gold_A[i][gold_class] = 1
        data.append(list())
        for a in range(num_annotators):
            spammer = np.random.binomial(1, 1 - gold_T[a])
            if spammer:
                # annotator a is a spammer
                data[-1].append((a, np.random.multinomial(1, gold_B[a]).argmax()))
            else:
                # annotator a is not a spammer
                data[-1].append((a, gold_class ))
            
    return data, gold_A, gold_B, gold_T

def EM(data, num_classes, num_annotators, num_iters = 20, smoothing = True):
    num_datapoints = len(data)
    
    # Random initialization
    # Warning: each row must sum to 1 in the two amtrices
    A = np.random.rand(num_datapoints, num_classes)
    A /= A.sum(axis=1, keepdims=True)

    B = np.random.rand(num_annotators, num_classes)
    B /= B.sum(axis=1, keepdims=True)

    T = np.random.rand(num_annotators)

    eta_1 = np.empty(num_annotators)
    eta_2 = np.empty(num_annotators)

    # Approximate posterior q
    # we need one vector per annotation
    # no need to initialise because we start with the E step
    q = [[np.empty((num_classes,2)) for _ in annotations] for annotations in data]

    elbos = []
    for _ in range(num_iters):
        e = 0
        # E step
        # update distribution q
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            for (a, c), item_q in zip(item_annotations, item_qs):
                # WARNING: inplace update
                item_q[:, 0] = A[item_idx] * (1 - T[a]) * B[a, c]
                item_q[:, 1] = A[item_idx] * T[a] * (c == np.arange(num_classes))
                s = ((1 - T[a]) * B[a, c]) + (A[item_idx, c]* T[a])
                item_q /= s
                t1 = (item_q[:, 0] * np.log(A[item_idx]* (1 - T[a]) * B[a, c] + 1e-10) + item_q[:, 1] * np.log(A[item_idx] * T[a] * (c == np.arange(num_classes)) + 1e-10)).sum()
                entropy = - (item_q * np.log(item_q + 1e-10)).sum()
                e += t1 + entropy 
        elbos.append(e)

        # M step
        # update model parameters
        smoothing_value = 0.1 / num_classes
        B.fill(0.)
        eta_1.fill(0.)
        eta_2.fill(0.)
        # eta_1 = np.zeros(num_annotators)
        # eta_2 = np.zeros(num_annotators)
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            # start with A, we update one item at a time
            # WARNING: inplace update
            # A[item_idx] = np.sum(item_qs, axis=0)

            A[item_idx] = np.sum(item_qs, axis = (0,2))
            if(smoothing):
                A[item_idx] = (A[item_idx]+smoothing_value) / (A[item_idx]+smoothing_value).sum()
            else:
                A[item_idx] /= A[item_idx].sum()
            # A[item_idx] = (A[item_idx]+smoothing) / (A[item_idx]+smoothing).sum()


            for (a, c), item_q in zip(item_annotations, item_qs):
                # c is the annotation made by annotator c in data item item_idx
                # item q is the vector of probabilities for the item
                eta_1[a] += item_q[:,1].sum()
                eta_2[a] += item_q[:,0].sum()

                B[a, c] += item_q[:,0].sum()

            
        # B renormalization
        # T update
        if(smoothing):
            B = (B+smoothing_value) / ((B+smoothing_value).sum(axis=1, keepdims=True))
            T = (eta_1 / eta_2 +smoothing_value) / (eta_1 / eta_2 + 1 + smoothing_value)
        else:
            B /= B.sum(axis=1, keepdims=True)
            T = (eta_1 / eta_2) / (eta_1 / eta_2 + 1 )

    return A, B, T, elbos

def EM_logspace(data, num_classes, num_annotators, num_iters=20, smoothing=True):
    num_datapoints = len(data)
    
    # Random initialization
    # Warning: each row must sum to 1 in the two amtrices
    A = np.random.rand(num_datapoints, num_classes)
    A /= A.sum(axis=1, keepdims=True)
    A = np.log(A)

    B = np.random.rand(num_annotators, num_classes)
    B /= B.sum(axis=1, keepdims=True)
    B = np.log(B)

    T = np.random.rand(num_annotators)
    T = np.log(T)

    eta_1 = np.empty(num_annotators)
    eta_2 = np.empty(num_annotators)
    # Approximate posterior q
    # we need one vector per annotation
    # no need to initialise because we start with the E step
    q = [[np.empty((num_classes,2)) for _ in annotations] for annotations in data]

    elbos = []
    for _ in range(num_iters):

        # E step
        # update distribution q
        e = 0
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            for (a, c), item_q in zip(item_annotations, item_qs):
                # WARNING: inplace update
                item_q[:, 0] = A[item_idx] + np.log1p(- np.exp(T[a]) + 1e-15) + B[a, c] # added a constant to avoid log(0)
                item_q[:, 1] = (A[item_idx] + T[a]) + np.log((c == np.arange(num_classes)) + 1e-15) # added a constant to avoid log(0)
                s = np.logaddexp(np.log1p(- np.exp(T[a])+ 1e-15) + B[a, c] , A[item_idx][c]+ T[a])
                item_q -= s
                entropy = - (item_q * np.exp(item_q)).sum()
                t1 = (np.exp(item_q[:, 0]) * (A[item_idx]+ np.log1p(- np.exp(T[a]) + 1e-15) + B[a, c]) + np.exp(item_q[:, 1]) * (A[item_idx] + T[a] + np.log((c == np.arange(num_classes))+ 1e-15))).sum()
                e += t1 + entropy
    
        elbos.append(e)
        # M step
        # update model parameters
        smoothing_value = 0.1 / num_classes
        B.fill(float("-inf"))
        eta_1.fill(float("-inf"))
        eta_2.fill(float("-inf"))
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            # WARNING: inplace update

            A[item_idx] =  np.logaddexp.reduce(item_qs, axis = (0,2))
            if(smoothing):
                A[item_idx] = np.logaddexp(A[item_idx], np.log(smoothing_value)) - logsumexp(np.logaddexp(A[item_idx], np.log(smoothing_value)))
            else:
                A[item_idx] -= logsumexp(A[item_idx])

            for (a, c), item_q in zip(item_annotations, item_qs):
                # print("a and c", a, c)
                eta_1[a] = np.logaddexp(eta_1[a], logsumexp(item_q[:,1]))
                eta_2[a] = np.logaddexp(eta_2[a], logsumexp(item_q[:,0]))
                B[a, c] = np.logaddexp(B[a, c], logsumexp(item_q[:,0]))
            

        # B renormalization
        # T update
        if(smoothing):
            B = np.logaddexp(B, np.log(smoothing_value)) - logsumexp(np.logaddexp(B, np.log(smoothing_value)), axis=1, keepdims=True)
            T = np.logaddexp((eta_1 - eta_2), np.log(smoothing_value)) - np.logaddexp(softplus_np(eta_1 - eta_2), np.log(smoothing_value))
            # T = (eta_1 - eta_2) - softplus_np(eta_1 - eta_2)
        else:
            B -= logsumexp(B, axis=1, keepdims=True)
            T = (eta_1 - eta_2) - softplus_np(eta_1 - eta_2)

    return np.exp(A), np.exp(B), np.exp(T), elbos


def evaluate(data, num_classes, num_annotators, num_iters=20, smoothing=False, num_restarts=10, logspace=False):
    best_elbo = -float("inf")
    best_A = None
    best_B = None
    best_T = None
    best_elbos = None
    for i in range(num_restarts):
        if logspace:
            A, B, T, elbos = EM_logspace(data, num_classes, num_annotators, num_iters, smoothing)
        else:
            A, B, T, elbos = EM(data, num_classes, num_annotators, num_iters, smoothing)
        if elbos[-1] > best_elbo:
            best_elbo = elbos[-1]
            best_A = A
            best_B = B
            best_T = T
            best_elbos = elbos
    return best_A, best_B, best_T, best_elbos
