import pytest
import numpy as np
import decision_tree as tree

class TestCalcEntropy:
    def test_basic1(self):
        y = [0,0,1,1]
        assert tree.calc_entropy(y) == 1

    def test_basic2(self):
        y = [0,1,1,1]
        assert tree.calc_entropy(y) == pytest.approx(0.811, 1e-3)
    
    def test_all_zeros(self):
        y = [0,0,0,0]
        assert tree.calc_entropy(y) == 0

    def test_all_ones(self):
        y = [1,1,1,1]
        assert tree.calc_entropy(y) == 0

    def test_empty(self):
        y = []
        assert tree.calc_entropy(y) == 0

