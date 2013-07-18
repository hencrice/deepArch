# -*- coding: UTF-8 -*-
from numpy import sum as npsum, asmatrix, log, asarray, append
from Helper import transform1Dto2D

def courseraML_CostFunc(forwardWeightAllLayers, *args):
    r'''
    Vectorized/regulated cost function (described in the Coursera Stanford Machine Learning course) that computes the total cost over multiple inputs:

    .. math::

        \frac{1}{m}&\sum_{i=1}^{m}\sum_{k=1}^{K}[-y_k^i log(forwardPropThruAllLayers(x^i)_k)-(1-y_k^i) log(1-forwardPropThruAllLayers(x^i)_k)]\\
        &+\frac{\lambda}{2m}\sum^{allLayers} (weight^2~if~weight~is~NOT~multiplied~with~any~bias~unit~otherwise~0)\\

    where m=the number of inputs; K=the number of labels in targets; y=a single target array; x=a single input array.

    :param forwardWeightAllLayers: A flatten array contains all forward weights from input layer to the last hidden layer before output layer.
    :param *args: Must in the following order:

        **inputArr2D**: 1 training example per row.

        **targets**: The number of labels must match the number of units in output layer.

        **weightDecayParam**: For model complexity regulation.

        **nn**: An instance of class FeedforwardNeuNet.

    :returns: A scalar representing the cost of current input using forwardWeightAllLayers.
    '''
    inputArr2D, targets, weightDecayParam, nn = args
    startIndex, weights2D = 0, []
    for ly in nn.layersExOutputLy: # update all forward weights
        newWeight = transform1Dto2D(forwardWeightAllLayers[startIndex:startIndex + ly._NnLayer__forwardWeight.size], *ly._NnLayer__forwardWeight.shape)
        ly._NnLayer__forwardWeight = asmatrix(newWeight)
        startIndex += ly._NnLayer__forwardWeight.size
        weights2D.append(newWeight)
    output = asarray(nn.forwardPropogateAllInput(inputArr2D))
    assert output.shape[1] == targets.shape[1], 'dimension mismatch in output layer'
    return 1.0 / targets.shape[0] * (npsum(-targets * log(output) - (1 - targets) * log(1 - output)) + weightDecayParam / 2.0 * npsum(npsum(w[:-1] ** 2) for w in weights2D)) # exclude weights for bias unit with [:-1]

def courseraML_CostFuncGrad(forwardWeightAllLayers, *args):
    r'''
    Vectorized/regulated implementation that computes the partial derivatives of the :func:`courseraML_CostFunc` over each weight in forwardWeightAllLayers:

    .. math::

        \frac{\partial~courseraMLCostFunc()}{\partial~weight_{ij}^{(layer~l)}}

    :param forwardWeightAllLayers: A flatten array contains all forward weights from input layer to the last hidden layer before output layer.
    :param *args: Must in the following order:

        **inputArr2D**: 1 training example per row.

        **targets**: The number of labels must match the number of units in output layer.

        **weightDecayParam**: For model complexity regulation.

        **nn**: An instance of class FeedforwardNeuNet.

    :returns: A flatten array represents the partial derivatives of the :func:`courseraML_CostFunc` over each weight in forwardWeightAllLayers.
    '''
    _, targets, weightDecayParam, nn = args # no need to use forwardWeightAllLayers cause ly._NnLayer__forwardWeight will be updated in courseraML_CostFunc(), which will be used in FeedforwardNeuNet.train() together with courseraML_CostFuncGrad()
    costGradAllLyOutToIn = [ly._NnLayer__forwardWeight for ly in reversed(nn.layersExOutputLy)] # each is a triangle^{l}(a matrix) on Courera ML L9, p8
    errDeltaArrNxtLv = (nn.outputs - targets)
    for n, ly in enumerate(reversed(nn.layersExOutputLy[1:])):
        costGradAllLyOutToIn[n] = ly.self2D.T * errDeltaArrNxtLv # Set m=(# of training examples), ly=layer below output layer => ly.self2D is a (m by (# of units, including bias)) matrix, and errDeltaArrNxtLv is a (m by (# of classes in output layer)) matrix. We originally have: outputLayerMatrix(which shares the same dimensions as errDeltaArrNxtLv)==ly.self2D*weightsMatrixBetweenOutputLyAndLyBelow(which shares the same dimensions as costGradAllLyOutToIn[n]), now we have: self.2D.T*outputLayerMatrix==self.2D.T*self.2D*weightsMatrixBetweenOutputLyAndLyBelow==weightsMatrixBetweenOutputLyAndLyBelow
        numOfExamples = ly.self2D.shape[0]
        costGradAllLyOutToIn[n] = 1.0 / numOfExamples * costGradAllLyOutToIn[n]
        costGradAllLyOutToIn[n][:-1] += weightDecayParam / numOfExamples * ly._NnLayer__forwardWeight[:-1] # add regularization but exclude weights for bias unit
        arr = asarray(ly.self2D)
        errDeltaArrNxtLv = asmatrix((asarray(errDeltaArrNxtLv * ly._NnLayer__forwardWeight.T) * arr * (1 - arr))[:, :-1]) # exclude bias unit
    ly = nn.layersExOutputLy[0] # no need to calculate errDeltaArrNxtLv for input layer, so separate it from for loop above
    costGradAllLyOutToIn[-1] = ly.self2D.T * errDeltaArrNxtLv # costGradAllLyOutToIn[-1] is the gradient of cost func over input layer's weights
    numOfExamples = ly.self2D.shape[0]
    costGradAllLyOutToIn[-1] = 1.0 / numOfExamples * costGradAllLyOutToIn[-1]
    costGradAllLyOutToIn[-1][:-1] += weightDecayParam / numOfExamples * ly._NnLayer__forwardWeight[:-1]
    flat = asarray(costGradAllLyOutToIn[-1])
    for f in reversed(costGradAllLyOutToIn[:-1]):
        flat = append(flat, asarray(f))
    return flat