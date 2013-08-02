# -*- coding: UTF-8 -*-
from numpy import sum as npsum, asmatrix, log, asarray, append, zeros, reshape

def courseraML_CostFunc(weightAllLayers, *args):
    r'''
    Vectorized/regulated cost function (described in the Coursera Stanford Machine Learning course) that computes the total cost over multiple inputs:

    .. math::

        &\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}[-y_k^i log(forwardThruAllLayers(x^i)_k)-(1-y_k^i) log(1-forwardThruAllLayers(x^i)_k)]\\
        &+\frac{\lambda}{2m}\sum^{allLayers~excludeAnyBias} (weight^2)\\

    where :math:`m` =the number of inputs; :math:`k` =the number of labels in targets; :math:`y` =a single target array; :math:`x` =a single input array.

    :param weightAllLayers: A flatten array contains all forward weights.
    :param *args: Must in the following order:

        **inputArr2D**: 1 training example per row.

        **targets**: Must be a 2D ndarray instead of matrix. And the number of labels must match the number of units in output layer.

        **weightDecayParam**: For model complexity regulation.

        **nn**: An instance of class FeedforwardNeuNet.

    :returns: A scalar representing the cost of current input using weightAllLayers.
    '''
    inputArr2D, targets, weightDecayParam, nn = args
    startIndex, weightsExcBias = 0, 0
    for ly in nn.layersExOutputLy:  # update all forward weights
        newWeight = reshape(weightAllLayers[startIndex:startIndex + ly.forwardWeight.size], ly.forwardWeight.shape)
        ly.forwardWeight = asmatrix(newWeight)
        startIndex += ly.forwardWeight.size
        weightsExcBias = append(weightsExcBias, newWeight[:-1])  # exclude weights for bias unit with [:-1]
    output = asarray(nn.forwardPropogateAllInput(inputArr2D))
    assert output.shape[1] == targets.shape[1], 'dimension mismatch in next layer'
    return 1.0 / targets.shape[0] * (npsum(-targets * log(output) - (1 - targets) * log(1 - output)) + weightDecayParam / 2.0 * npsum(weightsExcBias ** 2))

def courseraML_CostFuncGrad(weightAllLayers, *args):
    r'''
    Vectorized/regulated implementation that computes the partial derivatives of the :func:`courseraML_CostFunc` over each weight (but exclude weights multiplied by bias units) in weightAllLayers:

    .. math::

        \frac{\partial~courseraMLCostFunc()}{\partial~weight_{ij}^{(layer~l)}}

    :param weightAllLayers: A flatten array contains all forward weights.
    :param *args: Must in the following order:

        **inputArr2D**: 1 training example per row.

        **targets**: The number of labels must match the number of units in output layer.

        **weightDecayParam**: For model complexity regulation.

        **nn**: An instance of class FeedforwardNeuNet.

    :returns: A flatten array represents the partial derivatives of the :func:`courseraML_CostFunc` over each weight in weightAllLayers.
    '''
    _, targets, weightDecayParam, nn = args  # no need to use weightAllLayers cause ly.forwardWeight will be updated in courseraML_CostFunc(), which will be used in FeedforwardNeuNet.train() together with courseraML_CostFuncGrad()
    costGradAllLyOutToIn = []  # each is a triangle^{l}(a matrix) on Courera ML L9, p8
    numOfExamples = targets.shape[0] * 1.0
    errDeltaMatNxtLv = (nn.outputs - targets)
    for ly in reversed(nn.layersExOutputLy[1:]):
        costGradAllLyOutToIn.append(ly.self2D.T * errDeltaMatNxtLv)  # Set m=(# of training examples), ly=layer below output layer => ly.self2D is a (m by (# of units, including bias)) matrix, and errDeltaMatNxtLv is a (m by (# of classes in output layer)) matrix. We originally have: outputLayerMatrix(which shares the same dimensions as errDeltaMatNxtLv)==ly.self2D*weightsMatrixBetweenOutputLyAndLyBelow(which shares the same dimensions as costGradAllLyOutToIn[n]), now we have: self.2D.T*outputLayerMatrix==self.2D.T*self.2D*weightsMatrixBetweenOutputLyAndLyBelow==weightsMatrixBetweenOutputLyAndLyBelow
        # (check sparseAutoencoder_2011new.pdf p8) cause all bias unit has value 1, so after the statement above, costGradAllLyOutToIn[-1][-1] == npsum(errDeltaMatNxtLv, 0)==the partial deri of bi at the botm of p7
        costGradAllLyOutToIn[-1] = 1 / numOfExamples * costGradAllLyOutToIn[-1]
        costGradAllLyOutToIn[-1][:-1] += weightDecayParam / numOfExamples * ly.forwardWeight[:-1]  # add regularization but exclude weights for bias unit
        arr = asarray(ly.self2D)
        errDeltaMatNxtLv = asmatrix((asarray(errDeltaMatNxtLv * ly.forwardWeight.T) * arr * (1 - arr))[:, :-1])  # exclude bias unit
    ly = nn.layersExOutputLy[0]  # no need to calculate errDeltaMatNxtLv for input layer, so separate it from for loop above
    costGradAllLyOutToIn.append(ly.self2D.T * errDeltaMatNxtLv)  # costGradAllLyOutToIn[-1] is the gradient of cost func over input layer's weights
    costGradAllLyOutToIn[-1] = 1 / numOfExamples * costGradAllLyOutToIn[-1]
    costGradAllLyOutToIn[-1][:-1] += weightDecayParam / numOfExamples * ly.forwardWeight[:-1]
    flat = asarray(costGradAllLyOutToIn[-1])
    for f in reversed(costGradAllLyOutToIn[:-1]):
        flat = append(flat, asarray(f))
    return flat

def sparse_CostFunc(weightAllLayers, *args):
    r'''
    Vectorized/regulated sparse cost function (described in the sparseae_reading.pdf on the `UFLDL Tutorial Exercise:Sparse_Autoencoder <http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder>`_) that computes the total cost over multiple inputs:

    .. math::
    
        &define: \hat{\rho}_j=\frac{1}{m}\sum_{i=1}^{m}[actvh_j(x^i)]\\
        &define: \sum_{j=1}^{h}KL(\rho||\hat{\rho}_j)=\sum_{j=1}^{f}\rho~log\frac{\rho}{\hat{\rho}_j}+(1-\rho)log\frac{1-\rho}{1-\hat{\rho}_j}\\
        &costFunction:~\frac{1}{m}\sum_{i=1}^{m}(0.5~||forwardThruAllLayers(x^i)-y^i||^2)]+\frac{\lambda}{2}\sum^{allLayers~excludeAnyBias} (weight^2)\\
        &+\beta\sum_{j=1}^{f}KL(\rho||\hat{\rho}_j)

    where :math:`\hat{\rho}_j` =average activation of hidden unit j; :math:`m` =the number of inputs; :math:`h` =number of hidden units exclude bias units; :math:`y^i` =a single target array; :math:`x^i` =a single input array; :math:`\beta` =this is sparseParam.

    :param weightAllLayers: A flatten array contains all forward weights.
    :param *args: Must in the following order:

        **inputArr2D**: 1 training example per row.

        **targets**: Must be a 2D ndarray instead of matrix. And the number of labels must match the number of units in output layer.

        **weightDecayParam**: For model complexity regulation.
        
        **sparsity**: Setting the sparsity of neural network.  
        
        **sparseParam**: For sparsity regulation.

        **nn**: An instance of class FeedforwardNeuNet.

    :returns: A scalar representing the cost of current input using weightAllLayers.
    '''
    inputArr2D, targets, weightDecayParam, sparsity, sparseParam, nn = args
    startIndex, weightsExcBias = 0, 0
    avgEx = 1.0 / targets.shape[0]
    for ly in nn.layersExOutputLy:  # update all forward weights
        newWeight = reshape(weightAllLayers[startIndex:startIndex + ly.forwardWeight.size], ly.forwardWeight.shape)
        ly.forwardWeight = asmatrix(newWeight)
        startIndex += ly.forwardWeight.size
        weightsExcBias = append(weightsExcBias, newWeight[:-1])  # exclude weights for bias unit with [:-1]
    output = asarray(nn.forwardPropogateAllInput(inputArr2D))
    assert output.shape[1] == targets.shape[1], 'dimension mismatch in next layer'
    avgActvArrAllLyAllEx = 0
    for ly in nn.layersExOutputLy[1:]:
        ly.avgActvArrAllEx = avgEx * npsum(ly.self2D[:, :-1], 0)
        avgActvArrAllLyAllEx = append(avgActvArrAllLyAllEx, ly.avgActvArrAllEx)  # not sure whether I should include bias here?
    avgActvArrAllLyAllEx = avgActvArrAllLyAllEx[1:]  # discard 0 at the beginning
    return avgEx * npsum(0.5 * (output - targets) ** 2) + 0.5 * weightDecayParam * npsum(weightsExcBias ** 2) + sparseParam * npsum(sparsity * log(sparsity / avgActvArrAllLyAllEx) + (1 - sparsity) * log((1 - sparsity) / (1 - avgActvArrAllLyAllEx)))
    # return avgEx * (npsum(-targets * log(output) - (1 - targets) * log(1 - output)) + weightDecayParam / 2.0 * npsum(weightsExcBias ** 2)) + sparseParam * npsum(sparsity * log(sparsity / avgActvArrAllLyAllEx) + (1 - sparsity) * log((1 - sparsity) / (1 - avgActvArrAllLyAllEx)))

def sparse_CostFuncGrad(weightAllLayers, *args):
    r'''
    Vectorized/regulated implementation that computes the partial derivatives of the :func:`sparse_CostFunc` over each weight (but exclude weights multiplied by bias units) in weightAllLayers:

    .. math::

        \frac{\partial~sparseCostFunc()}{\partial~weight_{ij}^{(layer~l)}}

    :param weightAllLayers: A flatten array contains all forward weights.
    :param *args: Must in the following order:

        **inputArr2D**: 1 training example per row.

        **targets**: The number of labels must match the number of units in output layer.

        **weightDecayParam**: For model complexity regulation.

        **sparsity**: Setting the sparsity of neural network.  
        
        **sparseParam**: For sparsity regulation.

        **nn**: An instance of class FeedforwardNeuNet.

    :returns: A flatten array represents the partial derivatives of the :func:`courseraML_CostFunc` over each weight in weightAllLayers.
    '''
    _, targets, weightDecayParam, sparsity, sparseParam, nn = args  # no need to use weightAllLayers cause ly.forwardWeight will be updated in courseraML_CostFunc(), which will be used in FeedforwardNeuNet.train() together with courseraML_CostFuncGrad()
    costGradAllLyOutToIn = []  # each is a triangle^{l}(a matrix) on Courera ML L9, p8
    numOfExamples = targets.shape[0] * 1.0
    arr = asarray(nn.outputs)
    errDeltaMatNxtLv = asmatrix(asarray((nn.outputs - targets)) * arr * (1 - arr))
    for ly in reversed(nn.layersExOutputLy[1:]):
        costGradAllLyOutToIn.append(ly.self2D.T * errDeltaMatNxtLv)  # Set m=(# of training examples), ly=layer below output layer => ly.self2D is a (m by (# of units, including bias)) matrix, and errDeltaMatNxtLv is a (m by (# of classes in output layer)) matrix. We originally have: outputLayerMatrix(which shares the same dimensions as errDeltaMatNxtLv)==ly.self2D*weightsMatrixBetweenOutputLyAndLyBelow(which shares the same dimensions as costGradAllLyOutToIn[n]), now we have: self.2D.T*outputLayerMatrix==self.2D.T*self.2D*weightsMatrixBetweenOutputLyAndLyBelow==weightsMatrixBetweenOutputLyAndLyBelow
        # (check sparseAutoencoder_2011new.pdf p8) cause all bias unit has value 1, so after the statement above, costGradAllLyOutToIn[-1][-1] == npsum(errDeltaMatNxtLv, 0)==the partial deri of bi at the botm of p7
        costGradAllLyOutToIn[-1] = 1 / numOfExamples * costGradAllLyOutToIn[-1]
        costGradAllLyOutToIn[-1][:-1] += weightDecayParam * ly.forwardWeight[:-1]  # add regularization but exclude weights for bias unit
        arr = asarray(ly.self2D)
        s = sparseParam * (-sparsity / ly.avgActvArrAllEx + (1 - sparsity) / (1 - ly.avgActvArrAllEx))
        s = append(s, zeros((s.shape[0], 1)), 1)  # append zeros beca instead of getting rid of bias units for every term(errDeltaMatNxtLv * ly.forwardWeight.T, s, arr in the following statement), I decided to dump them as the final step
        errDeltaMatNxtLv = asmatrix(asarray(errDeltaMatNxtLv * ly.forwardWeight.T + s) * arr * (1 - arr))[:, :-1]  # exclude bias unit
    ly = nn.layersExOutputLy[0]  # no need to calculate errDeltaMatNxtLv for input layer, so separate it from for loop above
    costGradAllLyOutToIn.append(ly.self2D.T * errDeltaMatNxtLv)  # costGradAllLyOutToIn[-1] is the gradient of cost func over input layer's weights
    costGradAllLyOutToIn[-1] = 1 / numOfExamples * costGradAllLyOutToIn[-1]
    costGradAllLyOutToIn[-1][:-1] += weightDecayParam / numOfExamples * ly.forwardWeight[:-1]
    flat = asarray(costGradAllLyOutToIn[-1])
    for f in reversed(costGradAllLyOutToIn[:-1]):
        flat = append(flat, asarray(f))
    return flat