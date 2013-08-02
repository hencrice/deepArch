# -*- coding: UTF-8 -*-
import sys
sys.path.append('../src')
from unittest import TestCase, main
from numpy import array, e as npe, matrix, sum as npsum, all as npall, asarray, ravel, append, select, identity, finfo
from numpy.random import rand, randint, seed
from numpy.testing import assert_array_almost_equal
from scipy.io import loadmat
from scipy.optimize import approx_fprime
from FeedforwardNeuNet import sigmoid, NnLayer, FeedforwardNeuNet
from CostFunc import courseraML_CostFunc, courseraML_CostFuncGrad, sparse_CostFunc, sparse_CostFuncGrad

_epsilon = finfo(float).eps ** 0.5
def check_grad(func, grad, x0, *args):
    """Check the correctness of a gradient function by comparing it against a
    (forward) finite-difference approximation of the gradient.

    Parameters
    ----------
    func: callable func(x0,*args)
        Function whose derivative is to be checked.
    grad: callable grad(x0, *args)
        Gradient of `func`.
    x0: ndarray
        Points to check `grad` against forward difference approximation of grad
        using `func`.
    args: \*args, optional
        Extra arguments passed to `func` and `grad`.
    :raises: AssertionError
    """
    assert_array_almost_equal(approx_fprime(x0, func, _epsilon, *args), grad(x0, *args))

class TestSigmoid(TestCase):
    def setUp(self):
        self.x, self.theta = array((1, 2, 3)), array((4, 5, 6))

    def test_sigmoid(self):
        result = sigmoid(self.theta * self.x)
        self.assertAlmostEqual(result[0], 1.0 / (1 + npe ** (-4)))
        self.assertAlmostEqual(result[1], 1.0 / (1 + npe ** (-10)))
        self.assertAlmostEqual(result[2], 1.0 / (1 + npe ** (-18)))

class TestNnLayer(TestCase):
    def setUp(self):
        self.hiddenL = NnLayer(sigmoid, 3, 1, 3)
        self.hiddenL[:-1] = array((0.1, 0.2, 0.3))
        self.hiddenL.updateForwardWeight(matrix(((0.4, 0.5, 0.6, 0.7), (0.8, 0.9, 1, 1.1), (1.2, 1.3, 1.4, 1.5))))

    def test_recvAndActvByOneInput(self):
        result = self.hiddenL.recvAndActvByOneInput(array((1, 1, 1)))
        self.assertAlmostEqual(result[0], 1.0 / (1 + npe ** (-npsum(array((0.4, 0.5, 0.6, 0.7)) * array((1, 1, 1, 1))))))
        self.assertAlmostEqual(result[1], 1.0 / (1 + npe ** (-npsum(array((0.8, 0.9 , 1, 1.1)) * array((1, 1, 1, 1))))))
        self.assertAlmostEqual(result[2], 1.0 / (1 + npe ** (-npsum(array((1.2, 1.3, 1.4, 1.5)) * array((1, 1, 1, 1))))))
        self.assertTrue(npall(self.hiddenL == array((1, 1, 1, 1))))

    def test_actvByAllInput(self):
        inputArr2D = array(((0.41, 0.51, 0.61), (0.73, 0.63, 0.53)))
        result = asarray(self.hiddenL.actvByAllInput(inputArr2D))
        assert_array_almost_equal(result[0], ravel(sigmoid(array((0.41, 0.51, 0.61, 1)) * matrix(((0.4, 0.5, 0.6, 0.7), (0.8, 0.9, 1, 1.1), (1.2, 1.3, 1.4, 1.5))).T)))
        assert_array_almost_equal(result[1], ravel(sigmoid(array((0.73, 0.63, 0.53, 1)) * matrix(((0.4, 0.5, 0.6, 0.7), (0.8, 0.9, 1, 1.1), (1.2, 1.3, 1.4, 1.5))).T)))

class TestNeuNet(TestCase):
    def setUp(self):
        projectRootPath = '/'.join(__file__.replace('\t', '/t').replace('\\', '/').split('/')[:-2]) + '/testDataSet/'
        forwardWeightAllLayers = loadmat(projectRootPath + 'Theta1.mat')['Theta1'], loadmat(projectRootPath + 'Theta2.mat')['Theta2']
        layersExOutputLy = (NnLayer(sigmoid, 400, 1, 25), NnLayer(sigmoid, 25, 1, 10))
        layersExOutputLy[0].updateForwardWeight(forwardWeightAllLayers[0])
        layersExOutputLy[1].updateForwardWeight(forwardWeightAllLayers[1])
        self.inputs = loadmat(projectRootPath + 'X.mat')['X']
        self.outputs = loadmat(projectRootPath + 'forwardPropOutputs.mat')['actualOutput']
        self.nn = FeedforwardNeuNet(layersExOutputLy, 1, 0.05, 1)

    def test_forwardPropogateOneInput(self):
        self.nn.forwardPropogateOneInput(self.inputs[0])
        assert_array_almost_equal(self.nn, self.outputs[0])

    def test_forwardPropogateAllInput(self):
        result = self.nn.forwardPropogateAllInput(self.inputs)
        assert_array_almost_equal(result, self.outputs)

class TestCourseraML_CostFunc(TestCase):
    def setUp(self):
        projectRootPath = '/'.join(__file__.replace('\t', '/t').replace('\\', '/').split('/')[:-2]) + '/testDataSet/'
        self.layersExOutputLy = (NnLayer(sigmoid, 400, 1, 25), NnLayer(sigmoid, 25, 1, 10))
        self.nn = FeedforwardNeuNet(self.layersExOutputLy, 1, 0.05, 1)
        self.inputs = loadmat(projectRootPath + 'X.mat')['X']
        self.forwardWeightAllLayers = append(loadmat(projectRootPath + 'Theta1.mat')['Theta1'].T, loadmat(projectRootPath + 'Theta2.mat')['Theta2'].T)  # need to use transpose cause I use row vector rather than col vector as neurons
        y = loadmat(projectRootPath + 'y.mat')['y']
        identityArr = identity(10)
        self.targets = select([y == 1, y == 2, y == 3, y == 4, y == 5, y == 6, y == 7, y == 8, y == 9, y == 10], [identityArr[0], identityArr[1], identityArr[2], identityArr[3], identityArr[4], identityArr[5], identityArr[6], identityArr[7], identityArr[8], identityArr[9]])

    # the following test cases assume float64 precision
    def test_courseraML_CostFuncMultiOutputNoRegul(self):
        result = courseraML_CostFunc(self.forwardWeightAllLayers, self.inputs, self.targets, 0, self.nn)
        self.assertAlmostEqual(result, 0.28762916498285923)

    def test_courseraML_CostFuncMultiOutputWithRegul(self):
        result = courseraML_CostFunc(self.forwardWeightAllLayers, self.inputs, self.targets, 1, self.nn)
        self.assertAlmostEqual(result, 0.38376985909092481)

class TestCourseraML_CostFuncGrad(TestCase):
    def setUp(self):
        seed(3)
        self.layersExOutputLy = (NnLayer(sigmoid, 4, 1, 3), NnLayer(sigmoid, 3, 1, 10))
        self.nn = FeedforwardNeuNet(self.layersExOutputLy, 0, 0.05, 1)
        self.inputs = rand(7, 4)
        identityArr = identity(10)
        y = randint(0, 10, 7)
        self.targets = array([identityArr[t] for t in y])

    def test_courseraML_CostFuncGradMultiOutputNoRegul(self):
        for x in xrange(3):
            weights1, weights2 = rand(3, 5), rand(10, 4)
            self.nn.layersExOutputLy[0].updateForwardWeight(weights1)
            self.nn.layersExOutputLy[1].updateForwardWeight(weights2)
            self.nn.forwardPropogateAllInput(self.inputs)
            # weightDecayParam must set to 0 in order to check my partial derivitives against numerical ones obtained by approx_fprime()
            check_grad(courseraML_CostFunc, courseraML_CostFuncGrad, append(weights1, weights2), self.inputs, self.targets, 0, self.nn)

class TestSparse_CostFuncGrad(TestCase):
    def setUp(self):
        self.layersExOutputLy = (NnLayer(sigmoid, 4, 1, 3), NnLayer(sigmoid, 3, 1, 10))
        self.nn = FeedforwardNeuNet(self.layersExOutputLy, 0, 0.05, 1)
        self.inputs = rand(7, 4)
        identityArr = identity(10)
        y = randint(0, 10, 7)
        self.targets = array([identityArr[t] for t in y])

    def test_sparse_CostFuncGradMultiOutputNoWeitDecayNoSparse(self):
        for x in xrange(3):
            weights1, weights2 = rand(3, 5), rand(10, 4)
            self.nn.layersExOutputLy[0].updateForwardWeight(weights1)
            self.nn.layersExOutputLy[1].updateForwardWeight(weights2)
            self.nn.forwardPropogateAllInput(self.inputs)
            # weightDecayParam must set to 0 in order to check my partial derivitives against numerical ones obtained by approx_fprime()
            check_grad(sparse_CostFunc, sparse_CostFuncGrad, append(weights1, weights2), self.inputs, self.targets, 0, 0.01, 0, self.nn)

if __name__ == "__main__":
    main()