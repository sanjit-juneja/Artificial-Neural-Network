import numpy as np
import cv2
from typing import List, Tuple
import random
import math
from ANN import ANN
from ActivationFunctionFile import Activation_Type
from matplotlib import pyplot as plt

# red, green, yellow, blue, orange, white
colors = ((0,0,1),(0,1,0),(0,1,1),(1,0,0),(0,0.5,1),(1,1,1))

class GraphicPatternTester:

    def __init__(self):
        self.hex_point_list = None

    def generate_data(self, N:int, num_bins:int = 4, which_pattern: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if which_pattern == 1:
            return self.generate_hex_data(N,num_bins)
        if which_pattern == 2:
            return self.generate_dots_data(N, num_bins)
        else:
            return self.generate_spiral_data(N,num_bins)

    def generate_spiral_data(self, N:int, num_bins:int = 4) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pts = []
        one_hot_out = []
        for i in range(N):
            x = random.random()*20-10
            y = random.random()*20-10
            r = math.sqrt(pow(x,2)+pow(y,2))
            angle = math.atan2(y,x)
            bin = int((angle+math.pi)/(2*math.pi)*num_bins+r/3)%num_bins
            input = np.array([x,y])
            output = np.zeros([num_bins])
            output[bin] = 1
            pts.append(input)
            one_hot_out.append(output)

        return (pts,one_hot_out)


    def generate_wave_data(self, N:int, num_bins:int = 4) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pts = []
        one_hot_out = []
        for i in range(N):
            r = random.random()*20-10
            c = random.random()*20-10
            r2 = 0.75*math.cos(c)
            val = int(0.125*c+0.5*r+r2)
            bin = val%num_bins
            input = np.array([r,c])
            output = np.zeros([num_bins])
            output[bin] = 1
            pts.append(input)
            one_hot_out.append(output)

        return (pts,one_hot_out)

    def generate_hex_data(self, N: int, num_bins: int = 4) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pts = []
        one_hot_out = []
        if self.hex_point_list is None:
            self.make_hex_image(num_bins)
        for i in range(N):
            r = random.random() * 20 - 10
            c = random.random() * 20 - 10
            shortest_d_squared = 999
            bin = 0
            for hex_pt in self.hex_point_list:
                d_squared = pow(r-hex_pt[0],2)+pow(c-hex_pt[1],2)
                if d_squared<shortest_d_squared:
                    shortest_d_squared = d_squared
                    bin = hex_pt[2]
            input = np.array([r, c])
            output = np.zeros([num_bins])
            output[bin] = 1
            pts.append(input)
            one_hot_out.append(output)

        return (pts, one_hot_out)


    def make_hex_image(self, num_bins:int=4):
        self.hex_point_list:List[Tuple[float,float,int]]=[]
        for a in range(-10,10,3):
            offset = ((10 + a) % 6) / 3
            for b in range(-10,10,3):
                self.hex_point_list.append((a,b+offset,random.randint(0,num_bins-1)))


    def generate_dots_data(self,N:int, num_bins:int=4) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        pts = []
        one_hot_out = []
        threshold_squared = math.pow(7 * 2 * math.pi / (num_bins) * 0.3, 2)
        print(f"{threshold_squared=}")
        for i in range(N):
            x = random.random()*20-10
            y = random.random()*20-10
            angle = math.atan2(y, x)
            bin = int((angle+math.pi)/(2*math.pi)*num_bins)%num_bins
            nearest_dot_angle = ((bin+0.5)*math.pi*2)/num_bins-math.pi
            nearest_dot_x = 7*math.cos(nearest_dot_angle)
            nearest_dot_y = 7*math.sin(nearest_dot_angle)
            r_squared = math.pow(x-nearest_dot_x,2)+math.pow(y-nearest_dot_y,2)
            if i==0:
                print(f"({x:3.2f}, {y:3.2f}) -> ({nearest_dot_x:3.2f}, {nearest_dot_y:3.2f}) r^2 = {r_squared}")
                print(f"{bin=}")
                print(f"{angle=}\t{nearest_dot_angle=}")
            if r_squared<threshold_squared:
                bin = (bin+2)%num_bins
            input = np.array([x, y])
            output = np.zeros([num_bins])
            output[bin] = 1
            pts.append(input)
            one_hot_out.append(output)

        return (pts, one_hot_out)

    def draw_sample(self,image,loc,one_hot,N=-1,x_offset=0,y_offset=0):
        """
        Draws the first N points in "loc" in the color indexed by "one_hot" into the ndarray, "image." The
        upper corner of the image is drawn at (x_offset,y_offset). The resulting subimage is 100px x 100 px.
        :param image: the ndarray image into which we are drawing.
        :param loc: a list of coordinates, range (-10,10) x (-10,10)
        :param one_hot: a corresponding list of colors in a one-hot format (i.e., [0 0 1 0 0] means color 2
        :param N: The number of points to plot, or -1 for all of them.
        :param x_offset: the x-position of the upper left corner at which to draw
        :param y_offset: the y-position of the upper left corner at which to draw.
        :return: None
        """
        if N==-1:
            N = len(loc)
        N = min(len(loc),N)

        for i in range (N):
            image[y_offset+int(5*(loc[i][0]+10))][x_offset+int(5*(loc[i][1]+10))]=colors[np.argmax(one_hot[i])]


if __name__ == "__main__":

    st = GraphicPatternTester()
    NUM_BINS = 5
    WHICH_PATTERN = 0    #   0=spiral, 1 = hex, 2= dots

    # generate the data sets for training, validation and testing
    train_inp, train_outp = st.generate_data(N=2000, num_bins=NUM_BINS, which_pattern=WHICH_PATTERN)
    validate_inp, validate_outp = st.generate_data(N=100,num_bins=NUM_BINS, which_pattern=WHICH_PATTERN)
    test_in, test_out = st.generate_data(N=1000,num_bins= NUM_BINS, which_pattern=WHICH_PATTERN)

    #create the window for the graphics and draw the "training" solution data in the upper left corner.
    image = np.zeros([600,600,3],dtype=float)
    st.draw_sample(image, train_inp, train_outp)
    cv2.putText(image,"<----'Actual' colors",(110,50),cv2.FONT_HERSHEY_PLAIN,fontScale=1, color=(1,1,1))


    #create our neural network. The input layer is size 2 (x,y), and the output layer is size NUM_BINS - a one-hot
    #                             representation of the bins
    myAnn = ANN((2,20,10,10,NUM_BINS), activation_ids=(Activation_Type.IDENTITY,
                                                       Activation_Type.SIGMOID,
                                                       Activation_Type.SIGMOID,
                                                       Activation_Type.SIGMOID,
                                                       Activation_Type.SIGMOID))


    RUNS_PER_IMAGE = 25
    beta = 0.5
    gamma = 0.3
    rms_levels = []
    percent_correct_levels = []
    for j in range(24*RUNS_PER_IMAGE):
        num_correct = 0
        for i in range(len(train_inp)):
            myAnn.predict(train_inp[i])
            alpha = 30*(pow(math.e, -1*beta*i)) + gamma
            print(alpha)
            myAnn.backpropagate(train_outp[i], alpha)
        for i in range(len(validate_inp)):
            result=myAnn.predict(validate_inp[i])
            if np.argmax(validate_outp[i]) == np.argmax(result):
                 num_correct+=1
        percent_correct = num_correct / len(validate_inp) * 100
        if j%RUNS_PER_IMAGE == 0:
            output = []
            for i in range(len(train_inp)):
                output.append(myAnn.predict(train_inp[i]))
            st.draw_sample(image, train_inp, output, x_offset=int(100 * j / RUNS_PER_IMAGE) % 600,
                           y_offset=100+100*int((100*j/RUNS_PER_IMAGE)/600))
            cv2.putText(image,f"{int(percent_correct)}%",(int(100*j/RUNS_PER_IMAGE)%600+66,
                           100+100*int((100*j/RUNS_PER_IMAGE)/600)+91),
                           fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(0,0,0))
            cv2.putText(image, f"{int(percent_correct)}%", (int(100 * j / RUNS_PER_IMAGE) % 600 + 65,
                                                            100 + 100 * int((100 * j / RUNS_PER_IMAGE) / 600) + 90),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1, 1, 1))
            cv2.imshow("result", image)
            cv2.waitKey(1)
        rms = 0
        for k in range(20):
            rnd_index = random.randrange(0, len(train_inp))
            myAnn.predict(train_inp[rnd_index])
            rms+= myAnn.get_RMS_Error_for_output(train_outp[rnd_index])
        rms /= 20
        # print(f"{j}\t{rms=:3.4f}\t{percent_correct=:3.2f}%")
        rms_levels.append(rms)
        percent_correct_levels.append(percent_correct)

    myAnn.save("testANN.dat")



    predicted = []
    correct_tested = 0
    for i in range(len(test_in)):
        predicted.append(myAnn.predict(test_in[i]))
        if np.argmax(test_out[i]) == np.argmax(predicted[i]):
            correct_tested += 1
    percent_correct_tested = correct_tested/len(test_in)*100
    st.draw_sample(image, test_in, predicted, x_offset=0,y_offset=500)
    cv2.putText(image,f"<-- test results. ({int(percent_correct_tested)}%) Press any key to continue.",(120,550),
                fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(1,1,1))

    print (f"Test result: {percent_correct_tested:3.2f}%")
    cv2.imshow("result", image)
    myAnn.visualize_prediction()


    fig,(graph1,graph2) = plt.subplots(1,2)
    graph1.plot(rms_levels)
    graph1.set(xlabel= "cycles",ylabel="RMSE")
    graph1.set_ylim([0,1])
    graph2.plot(percent_correct_levels)
    graph2.set(xlabel="cycles",ylabel="% accuracy")
    graph2.set_ylim([0,100])
    fig.tight_layout()
    plt.show()
    cv2.destroyAllWindows()
