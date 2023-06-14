import numpy as np
from scipy.special import logsumexp

# to simplify, all annotators will annotate each data point once,
# but the data structure allows for anything else
def data_generator(num_classes, num_annotators, num_datapoints):
    # class-level abilities per annotator
    gold_B = np.empty((num_annotators, num_classes, num_classes))

    for a in range(num_annotators):
        for k in range(num_classes):
            gold_B[a, k] = np.random.dirichlet(np.ones(num_classes))

    # class prevalence
    pi = np.random.dirichlet(np.ones(num_classes))

    gold_A = np.empty((num_datapoints, num_classes))
    data = list()
    for i in range(num_datapoints):
        gold_A[i] = np.random.multinomial(1, pi)
        gold_class = gold_A[i].argmax()
        data.append(list())
        for a in range(num_annotators):
            data[-1].append((a, np.random.multinomial(1, gold_B[a][gold_class]).argmax()))
            
    return data, gold_A, gold_B

def EM(data, num_classes, num_annotators, num_iters=20):
    num_datapoints = len(data)
    
    # Random initialization
    # Warning: each row must sum to 1 in the two amtrices
    A = np.random.rand(num_datapoints, num_classes)
    A /= A.sum(axis=1, keepdims=True)

    B = np.random.rand(num_annotators, num_classes, num_classes)
    B /= B.sum(axis=2, keepdims=True)

    # Approximate posterior q
    # we need one vector per annotation
    # no need to initialise because we start with the E step
    q = [[np.empty(num_classes) for _ in annotations] for annotations in data]

    elbos = []
    for _ in range(num_iters):
        # E step
        # update distribution q
        e = 0
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            for (a, c), item_q in zip(item_annotations, item_qs):
                # WARNING: inplace update
                item_q[:] = A[item_idx] * B[a,:, c]
                item_q /= item_q.sum()
                entropy = -np.sum(item_q * np.log(item_q + 1e-10))
                t1 = (item_q * np.log(A[item_idx] * B[a, :, c] + 1e-10)).sum()
                e += t1 + entropy
        elbos.append(e)

        # M step
        # update model parameters
        B.fill(0.)
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            # start with A, we update one item at a time
            A[item_idx] = sum(item_qs)
            A[item_idx] /= A[item_idx].sum()

            for (a, c), item_q in zip(item_annotations, item_qs):
                B[a, :, c] += item_q

        # B renormalization
        B /= B.sum(axis=2, keepdims=True)

    return A, B, elbos


def EM_logspace(data, num_classes, num_annotators, num_iters=20):
    num_datapoints = len(data)
    
    # Random initialization
    # Warning: each row must sum to 1 in the two amtrices
    A = np.random.rand(num_datapoints, num_classes)
    A /= A.sum(axis=1, keepdims=True)
    A = np.log(A)

    B = np.random.rand(num_annotators, num_classes, num_classes)
    B /= B.sum(axis=2, keepdims=True)
    B = np.log(B)


    # Approximate posterior q
    # we need one vector per annotation
    # no need to initialise because we start with the E step
    q = [[np.empty(num_classes) for _ in annotations] for annotations in data]

    elbos = []
    for _ in range(num_iters):
        # E step
        # update distribution q
        e = 0
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            for (a, c), item_q in zip(item_annotations, item_qs):
                # WARNING: inplace update
                item_q[:] = A[item_idx] + B[a, :, c]
                item_q -= logsumexp(item_q)
                entropy = -np.sum(np.exp(item_q) * item_q)
                t1 = (np.exp(item_q) * (A[item_idx] + B[a, :, c])).sum()
                e += t1 + entropy
        elbos.append(e)

        # M step
        # update model parameters
        B.fill(float("-inf"))
        for item_idx, (item_annotations, item_qs) in enumerate(zip(data, q)):
            # start with A, we update one item at a time
            A[item_idx] = np.logaddexp.reduce(item_qs)
            A[item_idx] -= logsumexp(A[item_idx])

            for (a, c), item_q in zip(item_annotations, item_qs):
                B[a, :, c] = np.logaddexp(B[a, :, c], item_q)

        # B renormalization
        B -= logsumexp(B, axis=2, keepdims=True)

    return np.exp(A), np.exp(B), elbos

def evaluate(data, num_classes, num_annotators, num_iters=20, num_restarts=10, logspace=False):
    best_elbo = -float("inf")
    best_A = None
    best_B = None
    best_elbos = None
    for i in range(num_restarts):
        if logspace:
            A, B, elbos = EM_logspace(data, num_classes, num_annotators, num_iters)
        else:
            A, B, elbos = EM(data, num_classes, num_annotators, num_iters)
        if elbos[-1] > best_elbo:
            best_elbo = elbos[-1]
            best_A = A
            best_B = B
            best_elbos = elbos
    return best_A, best_B, best_elbos
