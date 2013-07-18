# -*- coding: UTF-8 -*-

from unittest import TestCase, main
from numpy import array
import sys
sys.path.append('../')
from Helper import transform1Dto2D

class TestHelper(TestCase):
    def test_transform1Dto2D(self):
        self.assertEqual(transform1Dto2D(array((1, 2, 3, 4, 5, 6, 7, 8)), 2, 4).shape, (2, 4))

if __name__ == "__main__":
    main()