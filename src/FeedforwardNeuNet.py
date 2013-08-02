# -*- coding: UTF-8 -*-
'''
This module contains 2 classes inherited from ndarray, NnLayer and NeuNet, which are the building
blocks of neural network. It also implements relating activation function such as sigmoid.
'''
from numpy import exp, ndarray, append, empty, ones, ravel, asmatrix, asarray, sum as npsum
from scipy.optimize import fmin_cg

def sigmoid(arr):
    '''
    Sigmoid function.
    '''
    return 1.0 / (1 + exp(-arr))

class NnLayer(ndarray):
    '''
    An instance of NnLayer class represents a single layer in neural network. The instance itself is an 1D ndarray used to store unit values (inputs + bias unit).
    '''
    def __new__(cls, activFunc, numOfUnit, biasUnitValue, numOfUnitNextLv):
        '''
        :param activFunc: callable *f(inputArr)* used to transform input to output.
        :param numOfUnit: number of units in this layer (bias unit is excluded).
        :param biasUnitValue: A constant. Note that bias unit will be the last unit in any layer.
        :param numOfUnitNextLv: Bias unit in the next level is excluded.

        >>> layer = NnLayer(sigmoid, 3, 1, 3)
        # the whole layer is treated as a column vector
        >>> layer[:-1] = (0.1, 0.2, 0.3)
        # in case you want to initialize unit values, using either tuple, list or numpy array.
        # Note that the last unit is bias unit so use -1 to exclude it.
        >>> layer.updateForwardWeight(((0.4, 0.5, 0.6, 0.7), (0.8, 0.9, 1, 1.1), (1.2, 1.3, 1.4, 1.5)))
        # in case you want to initialize forward weights.
        '''
        obj = append(empty(numOfUnit), biasUnitValue).view(cls)
        obj.__activFunc = activFunc
        obj.forwardWeight = asmatrix(empty((obj.size, numOfUnitNextLv)))
        return obj

#    __array_finalize__ is the only method that always sees new instances being created, it is the sensible place to fill in instance defaults for new object attributes, among other tasks.
#    def __array_finalize__(self, obj):
#        pass

    def recvAndActvByOneInput(self, inputArr):
        '''
        Given inputArr and activation function, compute the output. InputArr will also be stored as the unit values in this layer.

        :param inputArr: 1D numpy array.

        :returns: 1D numpy array representing the output.

        >>> layer = NnLayer(sigmoid, 3, 1, 3)
        >>> layer.updateForwardWeight(((0.4, 0.5, 0.6, 0.7), (0.8, 0.9, 1, 1.1), (1.2, 1.3, 1.4, 1.5)))
        # no need to provide input for bias unit
        >>> layer.recvAndActvByOneInput(array((0.1, 0.2, 0.3)))
        array([0.7349726, 0.840238, 0.90887704])
        >>> layer
        NnLayer([0.1, 0.2, 0.3, 1.])
        '''
        self[:-1] = inputArr
        return ravel(self.__activFunc(asarray(self) * self.forwardWeight))

    def actvByAllInput(self, inputArr2D):
        '''
        Given inputs as 2D array/matrix, computes the outputs as a numpy matrix. Unit values will not be updated by inputArr2D, instead, self.self2D will be updated.

        :param inputArr2D: 2D numpy array/matrix, each row contrains an input.

        :returns: Outputs as a numpy matrix.

        >>> layer = NnLayer(sigmoid, 3, 1, 3)
        >>> layer.updateForwardWeight(((0.4, 0.5, 0.6, 0.7), (0.8, 0.9, 1, 1.1), (1.2, 1.3, 1.4, 1.5)))
        >>> layer[:-1]=(0.23, 0.37, 0.28)
        >>> layer.actvByAllInput(array(((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))))
        matrix([[0.7349726 , 0.840238  , 0.90887704],
                [0.81305739, 0.92201176, 0.96982202]])
        >>> layer
        NnLayer([0.23, 0.37, 0.28, 1.])
        '''
        self.self2D = append(inputArr2D, ones((inputArr2D.shape[0], 1)) * self[-1], 1)  # self2D[-1] == bias
        return self.__activFunc(self.self2D * self.forwardWeight)

    def updateForwardWeight(self, newWeight):
        '''
        Modify forward propagation weights of a certain layer.

        :param newWeight: Can be either nested tuple/list or numpy matrix.
        '''
        newWeight = asmatrix(newWeight).T
        assert self.forwardWeight.shape == newWeight.shape, 'weight matrices dimension mismatch'
        self.forwardWeight = newWeight

class FeedforwardNeuNet(ndarray):
    '''
    An instance of this class represents a single neural network. The instance itself is an 1D numpy array served as the output layer.
    '''
    def __new__(cls, layersExOutputLy, weightDecayParam, sparsity, sparseParam):
        '''
        :param layersExOutputLy: A tuple that contains all layers except the output layer.
        :param weightDecayParam: Parameter that regulates network model complexity.
        :param sparsity: Parameter that sets the target value of the sparsity of the neural network.
        :param sparseParam: Parameter that regulates the sparsity.

        >>> layer0, layer1 = NnLayer(sigmoid, 3, 1, 2), NnLayer(sigmoid, 2, 1, 6)
        >>> layer0.updateForwardWeight(((0.11, 0.12, 0.13, 0.14), (0.15, 0.16, 0.17, 0.18)))
        >>> layer1.updateForwardWeight(((0.201, 0.202, 0.203), (0.204, 0.205, 0.206), (0.207, 0.208, 0.209), (0.21, 0.211, 0.212), (0.213, 0.214, 0.215), (0.216, 0.217, 0.218)))
        >>> nn = NeuNet((layers0, layer1), 1, 0.05, 1)
        '''
        obj = empty(layersExOutputLy[-1].forwardWeight.shape[1]).view(cls)
        obj.layersExOutputLy = layersExOutputLy
        obj.avgActvAllLyExOutputLyOverAllEx = None
        obj.__weightDecayParam = weightDecayParam
        obj.__sparsity = sparsity
        obj.__sparseParam = sparseParam
        return obj

    def train(self, inputArr2D, targets, costFunc, costFuncGrad, maxIter=100):
        '''
        This method will fit the weights of the neural network to the targets.
        :param inputArr2D: 1 input per row.
        :param targets: ground truth class label for each input
        :param costFunc: callable *f(paramToOptimize, \*arg)* that will be used as cost function.
        :param costFuncGrad: callable *f'(paramToOptimize, \*arg)* that will be used to compute partial derivative of cost function over each parameter in paramToOptimize.
        '''
        self.forwardPropogateAllInput(inputArr2D)  # perform forward propagation to set self.outputs
        avgEx = 1.0 / targets.shape[0]
        flatWeights = asarray(self.layersExOutputLy[0].forwardWeight)
        for ly in self.layersExOutputLy[1:]:
            ly.avgActvArrAllEx = avgEx * npsum(ly.self2D[:, :-1], 0)
            flatWeights = append(flatWeights, asarray(ly.forwardWeight))
        fmin_cg(costFunc, flatWeights, costFuncGrad, (inputArr2D, targets, self.__weightDecayParam, self.__sparsity, self.__sparseParam, self), maxiter=maxIter, full_output=True)  # fmin_cg calls grad before cost func

    def forwardPropogateOneInput(self, inputArr):
        '''
        Perform forward propagation using inputArr. Outputs will be stored as self.

        :param inputArr: 1D numpy array.

        :returns: D numpy array representing the output.

        >>> nn.forwardPropogateOneInput(array((0.17, 0.14, 0.71)))
        array([0.59976835, 0.60120776, 0.60264543, 0.60408132, 0.60551543, 0.60694771])
        >>> nn
        NeuNet([0.59976835, 0.60120776, 0.60264543, 0.60408132, 0.60551543, 0.60694771])
        '''
        hiddenLayer = self.layersExOutputLy[0].recvAndActvByOneInput(inputArr)
        for ly in self.layersExOutputLy[1:]:
            hiddenLayer = ly.recvAndActvByOneInput(hiddenLayer)
        assert hiddenLayer.size == self.size, 'dimension mismatch in output layer'
        self[:] = hiddenLayer
        return hiddenLayer

    def forwardPropogateAllInput(self, inputArr2D):
        '''
        Perform forward propagation on inputArr2D. Outputs will be stored as self.outputs.

        :param inputArr2D: numpy matrix/array, 1 training example per row.

        :returns: A numpy matrix as outputs.

        >>> nn[:]=array((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        >>> nn.forwardPropogateAllInput(array(((0.17, 0.14, 0.71), (0.11, 0.59, 0.327))))
        matrix([[0.59976835, 0.60120776, 0.60264543, 0.60408132, 0.60551543, 0.60694771],
                [0.59976835, 0.60120776, 0.60264543, 0.60408132, 0.60551543, 0.60694771]])
        >>> nn
        NeuNet([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        '''
        hiddenLayerAllOutput = self.layersExOutputLy[0].actvByAllInput(inputArr2D)
        for ly in self.layersExOutputLy[1:]:
            hiddenLayerAllOutput = ly.actvByAllInput(hiddenLayerAllOutput)
        assert hiddenLayerAllOutput.shape[1] == self.size, 'dimension mismatch in output layer'
        self.outputs = hiddenLayerAllOutput
        return hiddenLayerAllOutput