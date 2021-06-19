from ActivationFunctionFile import *
from typing import List, Tuple
import traceback
import numpy as np
import cv2
import math




class ANN:

    def __init__(self, layer_sizes:List[int], activation_ids:Tuple[int] = None):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        # set up and randomize the weight matrices between layers
        self.weight_matrices:List[np.ndarray] = []
        for i in range(0,len(self.layer_sizes)-1):
            self.weight_matrices.append(np.random.rand(layer_sizes[i]+1,layer_sizes[i+1]))

        # set up the list of application ids
        self.build_activation_functions_list(activation_ids)

        # set up lists to track intermediate values of a, and y_hat for each layer from run to run.
        self.a_values_for_layers: List[np.ndarray] = [None] * (self.num_layers)
        self.y_hats_for_layers:List[np.ndarray] = [None] * (self.num_layers)

    def build_activation_functions_list(self,activation_ids:Tuple[int]):
        """
        generate a list of activation ids that corresponds to the requested list of ids. If only one id is given, use all
        of that type. If no id is give, fill the list with SIGMOIDS.
        :param activation_ids: a list of activation ids
        :return: None
        """
        if activation_ids is None:
            activation_ids = Activation_Type.SIGMOID
        if isinstance(activation_ids, Activation_Type):
            activation_ids = [activation_ids] * self.num_layers
            activation_ids = tuple(activation_ids)
        if not isinstance(activation_ids, Tuple) or not isinstance(activation_ids[0], Activation_Type) or len(activation_ids) != self.num_layers:
            raise Exception(f"activations is in the wrong format. {activation_ids=}")
            traceback.print_tb()
        self.activation_functions = activation_ids

    def save(self,filename):
        """
        saves the state of this ANN to file, so that it can be recovered later.
        :param filename: the path to the file to write
        :return: None
        """
        # First, the number of layers
        output=f"{self.num_layers}\n"
        # Next, the number of nodes per layer
        for i in range(self.num_layers):
            output+=f"{self.layer_sizes[i]}\t"
        output+="\n"
        # Now the activation functions for each layer. (These are going to appear as integers)
        for i in range(self.num_layers):
            output += f"{self.activation_functions[i].name}\t"
        output += "\n"
        # Now we print out the weights.
        for i in range(self.num_layers-1):
            for j in range(self.weight_matrices[i].shape[0]):
                for k in range(self.weight_matrices[i].shape[1]):
                    if k>0:
                        output+="\t"
                    output+=f"{self.weight_matrices[i][j,k]}"
                output+="\n"

        #open the file and write to it
        fout = open(filename,"w")
        fout.write(output)
        fout.close()

    @classmethod
    def fromFile(cls,filename)->"ANN":
        """
        reads the ANN data stored in the given file and returns an instance of ANN based on the data found. This is a
        classmethod, which means that you call the class, ANN, instead of an object, and it makes an instance of the class.
        For instance, "myANN = ANN.fromFile('path_to_file.dat');"

        :param filename: the path to the file to open.
        :return: an ANN based on this data
        """
        fin = open(filename,'r')
        num_layers = int(fin.readline())
        layer_size_line = fin.readline()
        sizes = layer_size_line.split("\t")
        layer_sizes=[]
        for i in range(num_layers):
            layer_sizes.append(int(sizes[i]))
        activations_line = fin.readline()
        activations_split = activations_line.split("\t")
        activations = []
        for i in range(num_layers):
            activations.append(Activation_Type[activations_split[i]])
        activations = tuple(activations)
        weights = []
        for i in range (num_layers-1):
            weight = np.zeros([layer_sizes[i]+1,layer_sizes[i+1]])
            for j in range(weight.shape[0]):
                weight_line = fin.readline()
                weight_split = weight_line.split("\t")
                for k in range(weight.shape[1]):
                    weight[j][k] = float(weight_split[k])
            weights.append(weight)


        result = ANN(layer_sizes=layer_sizes, activation_ids=activations)
        result.weight_matrices = weights
        return result


    def predict(self, input_values: np.ndarray)-> np.ndarray:
        """
        Given an array of values to enter into the input layer, finds the corresponding output from this ANN. (Tracks
        intermediate inputs and outputs for hidden layers into class variables for later reference.)
        :param input_values: an array of values, the same size as the input layer.
        :return: an array of values, the same size as the output layer
        """
        #self.a_values_for_layers tracks the information RECEIVED BY EACH LAYER during this prediction. It is a list of
        #   1-d np.ndarrays, one ndarray per layer of the ANN.
        #   For the first layer, we are sending just one "input" signal into the dendrites, so this sum, a, is just the
        #        original input values.
        self.a_values_for_layers[0] = input_values

        #y_hats_for_layers tracks the information SENT OUT BY EACH LAYER during this prediction. It is a list of
        #   1-d np.ndarrays, one ndarray per layer of the ANN.
        self.y_hats_for_layers[0] = self.apply_activation_for_layer(layer_num=0, a_values=input_values)
        #-----------------------------------------------------------
        # TODO: Now you will be repeating this for all the other layers. As we move from one layer to the next,
        #  consider that the y-hat ("to axon") values for one layer plus a bias node become the x-values ("output_from_layer") for the
        #  next layer.
        #  There are three steps:
        #   1) Take the output (y_hat) from the previous layer and add a bias node to it to get the x_values for this layer.
        #   (Hint: there is a method in this class that does this.)
        #   2) Use matrix multiplication with the (N+1) x_values and the ((N+1) x M) weight matrix to get M a_values.
        #   3) Use the same idea as above to calculate the output of that next layer (y-hat), based on the input (a)
        #      you just calculated.
        # Note: The numbering of the layers and weight matrices can be confusing. The first time through this loop,
        #       you will be using the output (y-hat) from layer #0 and the #0 weight matrix to find the #1 input,
        #   and then the #1 output.

        #Now calculate the input into each subsequent layer and the output from that layer.
        for i in range(1, self.num_layers):
            self.a_values_for_layers[i] = np.matmul(self.add_bias_node(self.y_hats_for_layers[i-1]), self.weight_matrices[i-1])
            self.y_hats_for_layers[i] = self.apply_activation_for_layer(layer_num=i, a_values=self.a_values_for_layers[i])
            
        #-----------------------------------------------------------
        #return the output from the last layer.
        return self.y_hats_for_layers[-1]


    def calculate_errors(self,expected_output:np.ndarray)->List[np.ndarray]:
        """
        The first stage of backpropagation, we are measuring how much error there was as a measure of the input (a) to each layer.
        (Except for layer #0 - we assume the input was the real input!) This is going to be used to make corrections to
        the weight matrices. We start with the output layer, since that is the one we have information about what "should"
        have come out, and then we can use that information to figure out what "should" have happened at the previous layer,
        then the layer before that, etc. (Thus the "back" in "backpropagation.")
        :param expected_output: "y," the "actual" value that this ANN _should_ have output (y_hat) in the last predict() run.
        :return: a List of ndarrays - each ndarray is the amount of "error" in the input (a) to that layer.
        """
        errors = []

        #The error in the last level is found by:
        last_y_difference = self.y_hats_for_layers[-1] - expected_output
        derivative_of_activation_function = self.get_derivative_of_activation_for_layer_from_output(-1, self.y_hats_for_layers[-1])

        last_error = np.multiply(last_y_difference, derivative_of_activation_function) #(Note: cellwise multiplication, not matrix mult.)

        #Record this error in the "errors" list.
        errors.append(last_error) #this should wind up as the last element in this array, since it was the last layer.

        # TODO: loop backwards through the hidden layers. For each hidden layer, use the matrix that leads from this
        #       to the layer whose error you just found to calculate the error:
        #       1) apply (error_for_next_layer • weight_matrix^T) to get the difference at the "y-hat" level of this layer. This
        #          will be one longer than the number of nodes in this layer, because of the shape of the weight matrix.
        #       2) remove the last value, which corresponds to the bias.
        #       3) cellwise multiply that reduced error at the y_hat layer by the f_activation'(y_hat) for this layer to
        #          find the error for this layer.
        #       4) Be sure to record each layer's errors into the errors list, so they wind up in order.
        #   There is no error for the input layer, just the hidden ones and the output layer. (So these errors are always
        #   and only found for the input to the next layer after a weight matrix.)

        for i in range(2, self.num_layers):
            last_y_difference = self.strip_bias_node(last_error.dot(np.transpose(self.weight_matrices[1-i])))
            derivative_of_activation_function = self.get_derivative_of_activation_for_layer_from_output(-i, self.y_hats_for_layers[-i])
            last_error = np.multiply(last_y_difference, derivative_of_activation_function)
            errors.append(last_error)

        errors.reverse()
        #----------------------------------------
        return errors

    def update_weights(self, error_per_layer:List[np.ndarray], alpha:float = 0.02):
        """
        Uses the backpropagated errors sent from the weight matrices to each layer and the outputs (y_hat) from the previous
        layer that fed into that weight matrix to determine a correction to the weight matrix. This is multiplied by a
        factor, alpha, that determines the scale of the corrections and then used to change the weights.
        :param error_per_layer: a list of the error in input to each layer (after the input layer)
        :param alpha: the rate at which we wish to make the corrections
        :return: None
        postcondition: the weight matrices have been updated.
        """
        #TODO: for each weight matrix:
        #  1) Add the bias node to the y-hat from the layer feeding this matrix to get the x_values for the matrix.
        #     (Hint: there's a method in this class that does this.)
        #  2) Use the a error matrix you found earlier (error_per_layer) for the "a" level of the next layer (size M) and that list of
        #     x values (size N+1) to generate a correction matrix, which is ((N+1) x M).
        #  3) Update the value of the weight matrix by multipling this correction by alpha and subtracting it from the
        #     weight matrix.

        for i in range(0, self.num_layers - 1):
            x_values = self.add_bias_node(self.y_hats_for_layers[i])
            x_values = x_values.reshape([x_values.size, 1])
            error_per_layer[i] = error_per_layer[i].reshape([1, error_per_layer[i].size])
            self.weight_matrices[i] = np.subtract(self.weight_matrices[i], alpha * (x_values.dot(error_per_layer[i])))

        #----------------------------------

    def backpropagate(self, expected_output:np.ndarray, alpha:float = 0.02):
        """
        The overall backpropagation process for training the ANN, we walk backwards through the network to find the
        input error to each hidden and output layer, and use that to make a correction to the weight matrices.
        :param expected_output: what the previous prediction _should_ have output
        :param alpha: the factor by which to make the corrections. Larger is faster, but may easily overshoot.
        :return: None
        """
        errors_at_layers = self.calculate_errors(expected_output)
        self.update_weights(errors_at_layers,alpha)

    def get_RMS_Error_for_output(self,expected_output:np.ndarray)->float:
        """
        Used to determine how close the predicted output is to the expected output; finds the "root-mean-squared-error"
        (RMSE) between the output from the previous run of predict() and the "actual" value that it _should_ have
        predicted. This can be handy to graph the progress of our program.
        :param expected_output:  the "ground truth," "actual" value that this ANN should have predicted.
        :return: a float representing the amount of error - 0 means a perfect match.
        """
        diff = self.y_hats_for_layers[-1] - expected_output
        diff_squared = np.dot(diff,diff)
        mean_squared = diff_squared/(len(expected_output)-1)
        rms = math.sqrt(mean_squared)
        return rms

    def add_bias_node(self, y_hats: np.ndarray) -> np.ndarray:
        """
        Converts the y_hat array from one layer into the x array for the next layer by adding a 1 to the end of the array
        to represent the bias value.
        :param y_hats: an ndarray from one layer, of size N
        :return: the "x" ndarray for the following layer, of size N+1
        """
        return np.append(y_hats, 1)

    def strip_bias_node(self, array_to_strip:np.ndarray)->np.ndarray:
        """
        given an ndarray of size N, which includes information about a bias bit, remove the bias bit, so you get
        an ndarray of size N-1
        :param x_values: an N ndarray
        :return: y_hats - an N - 1 ndarray
        """
        return array_to_strip[:-1]

    def apply_activation_for_layer(self, layer_num:int, a_values:np.ndarray)->np.ndarray:
        """
        When you set up this ANN, you gave it a list of activation functions, one per layer. This applies that
        activation function (for the given layer) to the inputs sent to the layer of neurons to find the outputs sent
        out by that layer.
        :param layer_num: which number layer is this? (an integer)
        :param a_values: the summed up total signal sent into each neuron on this layer.
        :return: the output (y-hat) sent out by each neuron in this layer.
        """
        return ACTIVATION[self.activation_functions[layer_num]](a_values)

    def get_derivative_of_activation_for_layer_from_output(self, layer_num: int, y_hat_values:np.ndarray)->np.ndarray:
        """

        :param layer_num: the number of the layer in question
        :param y_hat_values: the most recent output from this layer of neurons
        :return: the derivative of the activation function at the most recent input to this layer of neurons.
        """
        return DERIVATIVE_FROM_ACTIVATION[self.activation_functions[layer_num]](y_hat_values)


    def visualize_prediction(self):
        """
        draws a picture of the state of this ANN during the most recent prediction run. Weight matrices are drawn as
        connecting lines in a gradient of colors: red means positive weight; blue means negative weight. "a" values for
        each node are in yellow, "y_hat" values in green.
        :return: None
        """
        CANVAS_WIDTH = 600
        CANVAS_HEIGHT = 600
        NODE_DIAM = 10
        canvas = np.ones([CANVAS_HEIGHT,CANVAS_WIDTH,3],dtype=float)*0.8
        pixels_per_layer = CANVAS_WIDTH/(len(self.layer_sizes))
        pixels_per_node = []
        for l in range(len(self.layer_sizes)):
            pixels_per_node.append(CANVAS_HEIGHT/(self.layer_sizes[l]+1))
        pixels_per_node[-1] = CANVAS_HEIGHT/self.layer_sizes[-1]
        for l in range(len(self.layer_sizes)):
            signal_sum_in = self.a_values_for_layers[l]
            for n in range(self.layer_sizes[l]+1):
                if n<self.layer_sizes[l]:
                    c = signal_sum_in[n].astype(float)
                    c = max(0,min(1,c))
                    col = (c,c,c)
                    c2 = self.y_hats_for_layers[l][n].astype(float)
                    c2 = max(0.0,min(1.0,c2))
                    col2 = (c2,c2,c2)

                else:
                    c = 1 # bias value
                    col = (1,1,1)
                    col2 = col

                # fill, based on signal into this neuron node
                cv2.circle(canvas, (int((l + 0.5) * pixels_per_layer), int((n + 0.5) * pixels_per_node[l])),
                           radius=int(NODE_DIAM / 2), color=col, thickness=-1)
                # stroke, 50% gray always.
                cv2.circle(canvas,(int((l+0.5)*pixels_per_layer),int((n+0.5)*pixels_per_node[l])),radius=int(NODE_DIAM/2),color=(0.5,0.5,0.5),thickness=1)

                # now draw weights
                if l<self.num_layers-1:
                    for n2 in range(self.layer_sizes[l+1]):
                        mat = self.weight_matrices[l]
                        col = (0.5+mat[n][n2]/2,0,0.5-mat[n][n2]/2) #color scale from 0 (blue) to 1 (red)

                        cv2.line(canvas,(int((l+0.5)*pixels_per_layer),int((n+0.5)*pixels_per_node[l])),
                                 (int((l+1.5)*pixels_per_layer)-20,int((n2+0.5)*pixels_per_node[l+1])),
                                 color = col, thickness=1)
                        cv2.line(canvas, (int((l + 1.5) * pixels_per_layer), int((n2 + 0.5) * pixels_per_node[l + 1])),
                                 ((int((l + 1.5) * pixels_per_layer) - 20, int((n2 + 0.5) * pixels_per_node[l + 1]))),
                                 color=(0, 0.4, 0.6), thickness=1)

                else:
                    cv2.line(canvas,(int((l+0.5)*pixels_per_layer),int((n+0.5)*pixels_per_node[l])),
                             (int((l+0.5)*pixels_per_layer)+40,int((n+0.5)*pixels_per_node[l])),
                             color = col2, thickness = 2)

                if n<self.layer_sizes[l]:
                    a = self.a_values_for_layers[l][n].astype(float)
                    y_hat = self.y_hats_for_layers[l][n].astype(float)
                    # cv2.putText(canvas,f"{a:1.2}",
                    #             (int((l+0.5)*pixels_per_layer-45),int((n+0.5)*pixels_per_node[l])-10),
                    #             fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color = (0,0.9,1))
                    cv2.line(canvas, (int((l + 0.5) * pixels_per_layer), int((n + 0.5) * pixels_per_node[l])),
                             ((int((l + 0.5) * pixels_per_layer) - 20, int((n + 0.5) * pixels_per_node[l]))),
                             color=(0, 0.4, 0.6), thickness=1)
                    # cv2.putText(canvas,f"{y_hat:1.2}",
                    #             (int((l+0.5)*pixels_per_layer+5),int((n+0.5)*pixels_per_node[l])+15),
                    #             fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color = (0.2,1,0.2))
        cv2.imshow("layers",canvas)



if __name__ == "__main__":
    a = ANN([3,8,6])
    input = np.array((1,2,3))
    # output = a.apply_layer_forward(input,0)
    # print (output)
    print("------")

    print(a.predict(np.array((0.25,0.75,0.33))))
    a.visualize_prediction()
    cv2.waitKey(0)
    print("------")
    for act in a.a_values_for_layers:
        print(act)

    print("------")
    for out in a.y_hats_for_layers:
        print(out)

