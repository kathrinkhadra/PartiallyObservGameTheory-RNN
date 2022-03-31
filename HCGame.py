#import airsim
import numpy as np
import pprint
import HCForwardGame
#import Drones
import KalmanFilter
import matplotlib.pyplot as plt
import modelRNN
import torch
from torch import nn

def GPU():
    ##GPU ETC. QUATSCH##
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    return device

def RNN_model():
    model = modelRNN.Model(input_size=6, output_size=8, hidden_dim=4, n_layers=2)
    #print(torch.load('models/models_One_Step/bestsofar/one_step_model.pt'))
    model.load_state_dict(torch.load('models/models_One_Step/bestsofar/one_step_model.pt'))#
    model.eval()
    return model

def plotting(traj_x_estimation,traj_y_estimation,traj_x_estimation_s,traj_y_estimation_s,traj_x_delay,traj_y_delay,traj_x_delay_s,traj_y_delay_s,delay):

    fig= plt.figure()
    for count,x in enumerate(traj_x_delay):
        plt.plot(traj_x_delay[count],traj_y_delay[count], ls='--', color='blue')
    for count,x in enumerate(traj_x_delay_s):
        plt.plot(traj_x_delay_s[count],traj_y_delay_s[count], color='red')
    plt.xlabel('x coordinate in m')
    plt.ylabel('y coordinate  in m')
    fig.savefig('fig/delay/delayedKalmanInput/second/alltraj/delay/'+str(delay)+'.png')   #save the figure to file
    plt.close(fig)

    fig1 = plt.figure()
    for count,x in enumerate(traj_x_estimation):
        plt.plot(traj_x_estimation[count],traj_y_estimation[count], ls='--', color='blue')
    for count,x in enumerate(traj_x_estimation_s):
        plt.plot(traj_x_estimation_s[count],traj_y_estimation_s[count], color='red')
    plt.xlabel('x coordinate in m')
    plt.ylabel('y coordinate in m')
    fig1.savefig('fig/delay/delayedKalmanInput/second/alltraj/estimation/'+str(delay)+'.png')   #save the figure to file
    plt.close(fig1)

def runHCGame(baseV, baseR, gammaVal, betaVal, maxT,two_Drones,delay,estimation,model,device):

    captureL = betaVal*baseR;
    dt = 0.01

    caughtData = []
    escapedData = []
    caught_delay=0
    caught_est=0

    theta_vec=np.arange(0,np.pi+(np.pi/8),(np.pi/8))

    rInit_vec=np.arange(2*captureL,11.5*captureL,captureL)

    figurecounter=0

    #file=open("fig/delay/delayedKalmanInput/second/newdata/caughtData"+str(delay)+".txt", "w+")
    #file1=open("fig/delay/delayedKalmanInput/second/newdata/caughtDataEstimation"+str(delay)+".txt", "w+")

    traj_x_estimation=[]
    traj_y_estimation=[]
    traj_x_estimation_s=[]
    traj_y_estimation_s=[]

    traj_x_delay=[]
    traj_y_delay=[]
    traj_x_delay_s=[]
    traj_y_delay_s=[]


    for rInit in rInit_vec:

        for theta in theta_vec:

            #open("RNNTrajectories/"+str(delay)+str(rInit)+str(theta)+".txt", "w+")
            #file1=open("ESTRNNTrajectories/EvaderReal/"+str(delay)+"/"+str(rInit)+"_"+str(theta)+".txt", "w+")
            #file2=open("ESTRNNTrajectories/Pursuer/"+str(delay)+"/"+str(rInit)+"_"+str(theta)+".txt", "w+")
            #file=open("Input_one_step_RNN/Pursuer/"+str(delay)+"_"+str(rInit)+"_"+str(theta)+".txt", "w+")

            #class constructor Kalman
            predict_Kal=KalmanFilter.Kalman(dt,None,None,None,[])

            #class constructor for the diff game
            twoDGame=HCForwardGame.HCForward([],[],[],[],rInit,theta,np.arange(0,maxT+dt,dt),gammaVal * baseV,baseV,baseR,betaVal * baseR,0, None, [], [], delay,[],[],0,[],[])

            print("begin round " + str(theta))

            #running the diff game for one theta and initial position
            twoDGame.HCForwardTimeGlobal(figurecounter,two_Drones,predict_Kal,model,device)

            #print("capturing " + str(twoDGame.caught))
            print("capturing " + str(twoDGame.caught_est))

            #saving the data
            xInit = rInit*np.cos(theta)
            yInit = rInit*np.sin(theta)

            figurecounter=figurecounter+1

            if False:
                for index,x in enumerate(twoDGame.xe_predicted):
                    #file.writelines(str(x)+","+str(twoDGame.ye_predicted[index])+"\n")
                    None
            #    file.close()

                for index,x in enumerate(twoDGame.x_e_est):
                    #file1.writelines(str(x)+","+str(twoDGame.y_e_est[index])+"\n")
                    None
            #    file1.close()

                for index,x in enumerate(twoDGame.x_p):
                    None
                    #file.writelines(str(x)+","+str(twoDGame.y_p[index])+"\n")
                #file.close()

            if False:

                if twoDGame.caught:
                    #file.writelines(str(theta)+": "+str([xInit, yInit])+" \n")
                    caughtData.append([xInit, yInit])
                    caught_delay=caught_delay+1
                    traj_x_delay_s.append(twoDGame.x_p)
                    traj_y_delay_s.append(twoDGame.y_p)
                else:
                    traj_x_delay.append(twoDGame.x_p)
                    traj_y_delay.append(twoDGame.y_p)

            if twoDGame.caught_est:
                caught_est=caught_est+1
                #traj_x_estimation_s.append(twoDGame.x_p_est)
                #traj_y_estimation_s.append(twoDGame.y_p_est)
            #else:
            #    if theta==theta_vec[len(theta_vec)-1]:
            #        y_p_est = [ -x for x in twoDGame.y_p_est]
            #        traj_y_estimation.append(y_p_est)
            #    else:
            #        traj_y_estimation.append(twoDGame.y_p_est)
            #    traj_x_estimation.append(twoDGame.x_p_est)





    #for count, y_p in enumerate(traj_y_estimation):
    #    if y_p[10]<0:
    #        print(count)
    #
    file=open("CatchSuccessOneStepModel/"+str(delay)+".txt", "w+")
    file.writelines(str(caught_est))
    file.close()
    #print(caught_est)


    #plotting(traj_x_estimation,traj_y_estimation,traj_x_estimation_s,traj_y_estimation_s,traj_x_delay,traj_y_delay,traj_x_delay_s,traj_y_delay_s,delay)
    #file.writelines(str(caught_delay))
    #file.close()
    #file1.writelines(str(caught_est))
    #file1.close()


if __name__ == "__main__":

    # connect to the AirSim simulator
    #two_Drones=Drones.Drones(airsim.MultirotorClient())
    #two_Drones.connecting()

    two_Drones=0
    # fly Drones
    #two_Drones.flying()

    # get state of Pursuer
    #two_Drones.get_state()

    # parameters
    baseVelocity=3 #v_p
    baseRadius=8
    gammaVal=0.8 #speed ratio  v_E/baseVelocity = gammaVal
    betaVal=0.2  # capture to minimum turning radius ratio
    maxT=60      # max time to go
    delay_array=[0,10,50,100,110,120,125,130,140,150,160,180,200] #
    #going through different thetas and initial positions
    device=GPU()
    model=RNN_model()
    for delay in delay_array:
    #delay=0
        runHCGame(baseVelocity, baseRadius, gammaVal, betaVal, maxT,two_Drones,delay,True,model,device)
