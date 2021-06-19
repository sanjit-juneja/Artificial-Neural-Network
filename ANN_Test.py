import unittest
from ANN import ANN
import numpy as np
from ActivationFunctionFile import Activation_Type

class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.myANN = ANN.fromFile("testingANN.dat")

    def test_1_first_activation(self):
        print("test 1")
        self.myANN.predict(np.array((0.8, 0.6, 0.4, 0.2, 0)))
        print(self.myANN.y_hats_for_layers[0])
        expected1 = np.array([0.68997448, 0.64565631, 0.59868766, 0.549834,   0.5])
        np.testing.assert_almost_equal(actual=self.myANN.y_hats_for_layers[0], desired=expected1, decimal= 5)


    def test_2_a_and_y_hats(self):
        print("test 2")
        self.myANN.predict(np.array((1.0,-0.5,0.0,-0.5,-1.0)))
        expected_as = [np.array([1., -0.5, 0., -0.5, -1.]),
                       np.array([1.67219403, 1.98473154, 1.39369625, 1.54398201, 1.94811212, 2.06485024, 2.02812228, 1.60509281]),
                       np.array([4.32923312, 3.6118534, 2.78584314, 2.87044873, 2.55395664, 3.23304541]),
                       np.array([3.73886022, 2.91403626, 4.59257246])]
        expected_y_hats = [np.array([0.73105858, 0.37754067, 0.5       , 0.37754067, 0.26894142]),
                           np.array([0.84186812, 0.87918464, 0.80118167, 0.82404285, 0.87524064, 0.88743957, 0.88371826, 0.83272897]),
                           np.array([0.98699374, 0.97370817, 0.941906  , 0.94636613, 0.92783888, 0.96205907]),
                           np.array([0.97677122, 0.94853595, 0.98997475])]

        for i in range(len(expected_as)):
            np.testing.assert_almost_equal(actual=self.myANN.a_values_for_layers[i], desired= expected_as[i], decimal = 5)
        for i in range(len(expected_y_hats)):
            np.testing.assert_almost_equal(actual=self.myANN.y_hats_for_layers[i], desired=expected_y_hats[i],decimal =5)

    def test_3_prediction(self):
        print("test 3")
        output = self.myANN.predict(np.array((-0.5,-0.25,0,0.25,0.5)))
        print(f"{output=}")
        expected3 = np.array((0.97687363, 0.94875073, 0.99003326))
        np.testing.assert_almost_equal(actual=output, desired=expected3, decimal = 5)


    def test_4_error_in_last_level(self):
        print("test 4")
        self.myANN.predict(np.array((3,4,5,6,7)))
        actual_y = np.array((1,0,0))
        errs = self.myANN.calculate_errors(actual_y)
        expected4 = np.array([-0.00049007,  0.04480772,  0.00936937])
        np.testing.assert_almost_equal(actual= errs[-1], desired = expected4, decimal = 5)

    def test_5_backpropagated_error_levels(self):
        print("test 5")
        self.myANN.predict(np.array((-0.8,-0.4,0.0,0.4,0.8)))
        actual_y = np.array((0.33,0.66,0.99))
        errs = self.myANN.calculate_errors(actual_y)
        expected5 = [np.array([9.98982418e-05, 9.80256732e-05, 2.85118269e-04, 1.30073861e-04, 1.96744536e-04, 2.01665489e-04, 1.27920832e-04, 2.22612083e-04]),
                     np.array([0.00013092, 0.00034067, 0.00054035, 0.00087581, 0.00134771, 0.0005307 ]),
                     np.array([1.46178591e-02, 1.40439971e-02, 2.86783652e-07])]
        for i in range(len(errs)):
            np.testing.assert_almost_equal(actual = errs[i], desired= expected5[i], decimal = 5)


    def test_6_new_weights_after_learning_once(self):
        print("test 6")
        output=self.myANN.predict(np.array((1,1,1,1,1)))
        actual_y = np.array((3,9,-5))
        self.myANN.backpropagate(expected_output=actual_y,alpha=0.05)
        expected6 = [np.array([[0.85073197, 0.96268444, 0.68371394, 0.62047001, 0.6704864 , 0.22771786, 0.11057179, 0.31151796],
                               [0.47249964, 0.50003416, 0.34801007, 0.27623196, 0.97785393, 0.51606713, 0.31444752, 0.43208772],
                               [0.49555375, 0.44620003, 0.88029283, 0.54262062, 0.53151932, 0.93082745, 0.58362098, 0.93961588],
                               [0.98687875, 0.62226481, 0.5453203 , 0.75350673, 0.96538022, 0.66656106, 0.77838039, 0.20575183],
                               [0.5336755 , 0.45880161, 0.19041776, 0.58891099, 0.13019126, 0.04723732, 0.96361536, 0.24338234],
                               [0.10806023, 0.51079105, 0.06550235, 0.27200887, 0.42367566, 0.97396954, 0.98385222, 0.60149278]]),
                     np.array([[0.99670001, 0.26399186, 0.22647327, 0.07071475, 0.21237145, 0.21307201],
                               [0.11310201, 0.4448973 , 0.08090161, 0.20797052, 0.01739302, 0.87906721],
                               [0.17769233, 0.57699804, 0.17294769, 0.13904436, 0.70209774, 0.61997532],
                               [0.23733946, 0.51773084, 0.33712434, 0.24816167, 0.01766538, 0.69639125],
                               [0.64265049, 0.77521795, 0.02821887, 0.52710068, 0.72629572, 0.02293881],
                               [0.69339194, 0.73353213, 0.92553311, 0.32360928, 0.63981155, 0.32197181],
                               [0.43742135, 0.07140335, 0.57608861, 0.83785379, 0.44198822, 0.39817173],
                               [0.88063541, 0.6163798 , 0.81909518, 0.37181074, 0.19842818, 0.50530273],
                               [0.75550562, 0.20532108, 0.07348293, 0.51581303, 0.03049878, 0.13222301]]),
                     np.array([[0.33796321, 0.42729125, 0.77881917],
                               [0.27555815, 0.69664077, 0.82730054],
                               [0.19181357, 0.55186007, 0.52973463],
                               [0.84570642, 0.41976742, 0.89171042],
                               [0.49208731, 0.96648969, 0.78328366],
                               [0.965716  , 0.07654822, 0.39422035],
                               [0.78494164, 0.05195361, 0.55066411]])]



        for i in range(len(expected6)):
            np.testing.assert_almost_equal(actual=self.myANN.weight_matrices[i],desired=expected6[i],decimal = 5)


    def test_7_new_prediction_after_learning_once(self):
        print("test 7")
        output2= self.myANN.predict(np.array((-1.25,-8,2.25,1.5,4.75)))
        actual_y = np.array((0.95, 0.1, 0.50))
        self.myANN.backpropagate(expected_output=actual_y,alpha=0.1)
        expected7 = np.array([0.97711697, 0.9479864 , 0.99014304])
        np.testing.assert_almost_equal(actual=self.myANN.predict(np.array((-1.25,-8,2.25,1.5,4.75))),desired=expected7,decimal=5)

if __name__ == '__main__':
    unittest.main()
