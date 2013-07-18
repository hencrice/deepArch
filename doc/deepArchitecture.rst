.. _deepArchitecture:

Deep Architectures
==================
Various types of deep architectures.

Feedforward Neural Network Structure
------------------------------------
.. image:: https://lh3.googleusercontent.com/-jxMPh0i6q5M/UeZFRAWHtrI/AAAAAAAAGpM/Y888-Vz2ri8/w987-h549-no/FeedforwardNNStruct.jpg

.. |addBias| image:: https://lh3.googleusercontent.com/-ZoJswguVmxQ/UeZHe6Y8zeI/AAAAAAAAGpk/RDza9WEoHzk/w48-h44-no/addBias.png
	:width: 40px
	:height: 40px

.. |weights| image:: https://lh6.googleusercontent.com/-4RBDIb7_VRY/UeZHiRa8nrI/AAAAAAAAGps/XCWAa7E8vGA/w89-h51-no/weights.png
	:width: 60px
	:height: 30px

.. |save_icon| image:: https://lh6.googleusercontent.com/-dmBSipdmYWY/UeZHnDw6L7I/AAAAAAAAGp0/iNiHZykQ2Lc/s14-no/save_icon.gif
	:width: 20px
	:height: 20px

.. |sigmoid| image:: https://lh4.googleusercontent.com/-MC8-zpYp6aU/UeZHpFFwCkI/AAAAAAAAGp8/sGCCrSxa9hI/w73-h40-no/sigmoid.png
	:width: 60px
	:height: 30px

|addBias| : add bias to the input. |weights| : multiple the input by the corresponding weights of that layer. |save_icon| : save the result as the values of this layer. |sigmoid| : execute sigmoid function on the input element-wise.

Feedforward Neural Network Implementation
-----------------------------------------
.. currentmodule:: FeedforwardNeuNet

.. autofunction:: sigmoid
	
.. autoclass:: NnLayer
	:special-members:
	:members:

.. autoclass:: FeedforwardNeuNet
	:special-members:
	:members:

Details
+++++++
Each layer is treated like a column vector. However, internally, layers are implemented as 1D numpy array, which is rather a row vector. So when we produce the linear combination of inputs by multiplying weights, we actually do:
:math:`layer*forwardWeights.T` rather than :math:`forwardWeights*layer`.