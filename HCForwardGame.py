import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler


class HCForward:

    def __init__(self,x_e,y_e,x_p,y_p,rInit,thetaInit,time,v_E,v_P,minR,captureL,caught,theta_e, xe_predicted,ye_predicted,delay,x_e_est,y_e_est,caught_est,x_p_est,y_p_est): #[],[],[],[],rInit,thetaInit,np.arange(0,maxT+dt,dt),gammaVal * baseV,baseV,baseR,betaVal * baseR,0,None, [],[],None           dt = 0.01 Tmax
        self.x_e=x_e
        self.y_e=y_e
        self.x_p=x_p
        self.y_p=y_p
        self.rInit=rInit
        self.thetaInit=thetaInit
        self.time=time
        self.v_E =v_E
        self.v_P = v_P
        self.minR = minR
        self.captureL = captureL
        self.caught=caught
        self.theta_e=theta_e
        self.xe_predicted=xe_predicted
        self.ye_predicted=ye_predicted
        self.delay=delay
        self.x_e_est=x_e_est
        self.y_e_est=y_e_est
        self.caught_est=caught_est
        self.x_p_est=x_p_est
        self.y_p_est=y_p_est

    #initial position of evader and pursuer
    def initialization(self):
        self.x_e.append(self.rInit*np.cos(self.thetaInit))
        self.y_e.append(self.rInit*np.sin(self.thetaInit))

        self.xe_predicted.append(self.rInit*np.cos(self.thetaInit))
        self.ye_predicted.append(self.rInit*np.sin(self.thetaInit))

        self.x_e_est.append(self.rInit*np.cos(self.thetaInit))
        self.y_e_est.append(self.rInit*np.sin(self.thetaInit))

        self.x_p.append(0)
        self.y_p.append(0)

        self.x_p_est.append(0)
        self.y_p_est.append(0)

    #save trajectories as plots
    def plotting(self,figurecounter, mu_x, mu_y):
        #x_value,v_x_value,y_value,v_y_value=zip(*self.xe_predicted)
        #x_value,v_x_value,y_value,v_y_value=zip(*self.xe_predicted)
        t=np.linspace(0, 60, len(self.x_p_est))
        #t_predict=np.linspace(0, 60, len(self.x_p_est)-1)
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.plot3D(self.x_p_est,t, self.y_p_est,label='pursuer')
        ax.plot3D(self.x_e_est,t, self.y_e_est,ls='--', label='evader')
        #plt.plot(self.x_e, self.y_e, label='evader');
        #plt.plot(self.x_p, self.y_p,label='pursuer' );
        #plt.plot(x_value, y_value, ls='--', label='predicted');
        #plt.plot(self.x_p_est, self.y_p_est,label='pursuer' );
        #plt.plot(self.x_e_est, self.y_e_est, ls='--', label='evader');
        #plt.plot(self.xe_predicted, self.ye_predicted, ls='--', label='predicted');
        #plt.plot(self.xe_predicted,self.xe_predicted, label='evader_predicted');
        #plt.plot(mu_x,mu_y, ls='--', label='KF output')
        ax.set_xlabel('x axis in m')
        ax.set_ylabel('time in s')
        ax.set_zlabel('y axis in m')
        plt.legend(loc=4)
        title='Pursuer and Evader Trajectory with a delay of '+str(self.delay*0.01)+'s'
        fig.suptitle(title, fontsize=16)
        #fig.savefig('Testplots/'+str(figurecounter)+'est.png')
        fig.savefig('CatchSuccessOneStepModel/'+str(self.delay)+'/'+str(figurecounter)+'catch.png')
        #fig.savefig('fig/delay/delayedKalmanInput/second/traj/'+str(self.delay)+'/xpvsxe/'+str(figurecounter)+'.png')   #save the figure to file
        plt.close(fig)

        #fig = plt.figure()
        #ax = Axes3D(fig)

        #ax.plot3D(t,self.x_e_est, self.y_e_est,label='evader')
        #ax.plot3D(t,self.xe_predicted, self.ye_predicted,ls='--', label='predicted')
        #plt.plot(self.x_e_est, self.y_e_est, label='evader');
        #plt.plot(self.x_p_est, self.y_p_est,label='pursuer' );
        #plt.plot(self.xe_predicted, self.ye_predicted, ls='--', label='predicted');
        #plt.plot(self.xe_predicted,self.xe_predicted, label='evader_predicted');
        #plt.plot(self.x_p, self.y_p,label='pursuer' );
        #plt.plot(mu_x,mu_y, ls='--', label='KF output')
        #plt.legend(loc=4)
        #fig.savefig('Testplots/'+str(figurecounter)+'est.png')
        #fig.savefig('fig/delay/delayedKalmanInput/second/traj/'+str(self.delay)+'/est/'+str(figurecounter)+'.png')   #save the figure to file
        #plt.close(fig)


        #fig = plt.figure()
        #plt.plot(list(range(len(self.x_e))),self.x_e, label='evader');
        #plt.plot( list(range(len(x_value))), x_value, ls='--', label='predicted');
        #fig.savefig('fig/overtime/'+str(figurecounter)+'.png')
        #plt.close(fig)


    #calculate theta and next state of pursuer with a delay in information
    def add_delay(self,theta_p_del,dt,count):

        #factor_delay=2*np.random.randn(1,1) #delay

        #add delay
        #self.x_e[count]=self.x_e[count]+factor_delay
        #self.y_e[count]=self.y_e[count]+factor_delay

        #calculation of relative distance of pursuer and evader
        if len(self.x_e)<self.delay:
            xRel = (self.x_e[0] - self.x_p[count])*np.cos(theta_p_del) + (self.y_e[0] - self.y_p[count])*np.sin(theta_p_del)
            yRel = -(self.x_e[0] - self.x_p[count])*np.sin(theta_p_del) + (self.y_e[0] - self.y_p[count])*np.cos(theta_p_del)
        else:
            xRel = (self.x_e[count-self.delay] - self.x_p[count])*np.cos(theta_p_del) + (self.y_e[count-self.delay] - self.y_p[count])*np.sin(theta_p_del)
            yRel = -(self.x_e[count-self.delay] - self.x_p[count])*np.sin(theta_p_del) + (self.y_e[count-self.delay] - self.y_p[count])*np.cos(theta_p_del)


        #calculation of angular velocities
        if yRel == 0:
            if xRel < 0:
                phi = 1
            else:
                phi = 0
        else:
            phi = np.sign(yRel)

        theta_p_del = theta_p_del + (self.v_P/self.minR)*phi*dt

        #calculation of next state
        x_p_step=self.x_p[count] + self.v_P*np.cos(theta_p_del)*dt
        y_p_step=self.y_p[count] + self.v_P*np.sin(theta_p_del)*dt

        return x_p_step,y_p_step,theta_p_del

    #predicition Kalman
    def execution_Kalman(self,predict_Kal,count,theta_p):

        correction=[]
        dt = 0.01

        for i in range(count-self.delay,count):
            #print(1)
            predict_Kal.kf.predict()

            xRel=self.x_e_est[i+1]-self.x_e_est[i]
            yRel=self.y_e_est[i+1]-self.y_e_est[i]
            #xRel=self.x_e_est[i]-self.x_p_est[i]#self.x_e_est[count-1]
            #yRel=self.y_e_est[i]-self.y_p_est[i]

            #if np.absolute(yRel)<=np.absolute(xRel):
            #    factor=2
            #else:
            #    factor=0.5

            if np.absolute(yRel)<=np.absolute(xRel):
                if np.sign(xRel)==-1:
                    if np.sign(yRel)==-1: #forward
                        psi =9*np.pi/8
                    else:
                        psi =7*np.pi/8
                else: #backward
                    if np.sign(yRel)==-1:
                        psi =15*np.pi/8
                    else:
                        psi =np.pi/8
            else:
                if np.sign(yRel)==-1:#right
                    if np.sign(xRel)==-1:
                        psi =11*np.pi/8
                    else:
                        psi =13*np.pi/8
                else: #left
                    if np.sign(xRel)==-1:
                        psi =5*np.pi/8
                    else:
                        psi =3*np.pi/8

            #psi = theta_p + np.arctan2(np.sign(yRel), np.sign(xRel))#*factor
            #
            measurement=[self.xe_predicted[i] + (self.v_E)*np.cos(psi)*dt,self.ye_predicted[i] + (self.v_E)*np.sin(psi)*dt]
            #measurement=[self.x_e_est[i+1],self.y_e_est[i+1]]
            #print([self.xe_predicted[i],self.ye_predicted[i]])
            #print(measurement)

            predict_Kal.kf.update(measurement)

            x_value=predict_Kal.kf.x[0]
            y_value=predict_Kal.kf.x[2]
            #print('-------------------')

            if i<len(self.xe_predicted)-1:
                self.xe_predicted[i+1]=x_value
                self.ye_predicted[i+1]=y_value
            else:
                self.xe_predicted.append(x_value)
                self.ye_predicted.append(y_value)
            #predict_Kal.kf.P=np.absolute(predict_Kal.kf.P)
            #print(predict_Kal.kf)
            #print(predict_Kal.kf)
            #print('-------------------------------------')
            #print(self.xe_predicted)
            #self.xe_predicted.append(x_value)
            #self.xe_predicted.append(y_value)


        if False:
            if self.delay<100:
                for i in range (self.delay,2*self.delay):
                    measurement=[self.x_e_est[count-i],self.y_e_est[count-i]]
                    correction.append(measurement)
            else:
                for i in range (self.delay,self.delay+100):
                    measurement=[self.x_e_est[count-i],self.y_e_est[count-i]]
                    correction.append(measurement)

            correction=np.flip(correction, 0)
            (mu,cov)= predict_Kal.kf.batch_filter(correction)

        ##################################################################
        #smoother step is not working
        #(xs, Ps, Ks) = predict_Kal.kf.rts_smoother(mu,cov)

##############################################
        #predict_Kal.kf.predict()
        #predict_Kal.kf.update([self.x_e_est[count-self.delay],self.y_e_est[count-self.delay]])

        #for i in range(2,self.delay):
        #    predict_Kal.kf.predict()
###########################################################

        ##own adaptive filtering
        #for i in range (0,self.delay):
            #predict_Kal.kf.predict()
            #predict_Kal.kf.update(correction)
            ##x_value,v_x_value,y_value,v_y_value=zip(*predict_Kal.kf.x)
            ##self.xe_predicted.append(predict_Kal.kf.x[0])
            ##self.ye_predicted.append(predict_Kal.kf.x[2])
            ##print(len(self.xe_predicted))
            ##print(len(self.ye_predicted))
            ##print("after estimation")

        #self.xe_predicted.append(predict_Kal.kf.x[0])
        #self.ye_predicted.append(predict_Kal.kf.x[2])
        #predict_Kal.kf.predict()
        #correction=[int(self.x_e[count-1]),int(self.y_e[count-1])]
        #predict_Kal.kf.update(correction)

        #predict_Kal.kf.predict()
        #correction=[int(self.x_e[count]),int(self.y_e[count])]
        #predict_Kal.kf.update(correction)


        ##

        #smoother
        #self.xe_predicted=np.concatenate((self.xe_predicted, xs[2,0]), axis=None)
        #self.ye_predicted=np.concatenate((self.ye_predicted, xs[2,2]), axis=None)
        #return mu

    #calculate theta and next state of pursuer with a delay in information
    def pursuers_trajectory_with_estimation(self,theta_p_est,dt,count):

        #x_value,v_x_value,y_value,v_y_value=zip(*self.xe_predicted)

        #calculation of relative distance of pursuer and evader
        xRel =(self.xe_predicted[count] - self.x_p_est[count])*np.cos(theta_p_est) + (self.ye_predicted[count] - self.y_p_est[count])*np.sin(theta_p_est)
        yRel =-(self.xe_predicted[count] - self.x_p_est[count])*np.sin(theta_p_est) + (self.ye_predicted[count] - self.y_p_est[count])*np.cos(theta_p_est)


        #calculation of angular velocities
        if yRel == 0:
            if xRel < 0:
                phi = 1
            else:
                phi = 0
        else:
            phi = np.sign(yRel)

        theta_p_est = theta_p_est + (self.v_P/self.minR)*phi*dt

        #calculation of next state
        x_p_step=self.x_p_est[count] + self.v_P*np.cos(theta_p_est)*dt
        y_p_step=self.y_p_est[count] + self.v_P*np.sin(theta_p_est)*dt

        return x_p_step,y_p_step,theta_p_est

    def newcontroller(self,count,model,device,theta):
        #correction=[]
        dt = 0.01


        # RNN here
        #xRel=self.x_e_est[count]-self.x_p_est[count]#self.x_e_est[count-1]
        #yRel=self.y_e_est[count]-self.y_p_est[count]#self.y_e_est[count-1]

        #if np.absolute(yRel)<=np.absolute(xRel):
        #    if np.sign(xRel)==-1:
        #        if np.sign(yRel)==-1: #forward
        #            psi =9*np.pi/8
        #        else:
        #            psi =7*np.pi/8
        #    else: #backward
        #        if np.sign(yRel)==-1:
        #            psi =15*np.pi/8
        #        else:
        #            psi =np.pi/8
        #else:
        #    if np.sign(yRel)==-1:#right
        #        if np.sign(xRel)==-1:
        #            psi =11*np.pi/8
        #        else:
        #            psi =13*np.pi/8
        #    else: #left
        #        if np.sign(xRel)==-1:
        #            psi =5*np.pi/8
        #        else:
        #            psi =3*np.pi/8

        #print(psi)

        vx_e=(self.x_e_est[count-self.delay]-self.x_e_est[count-self.delay-1])/dt
        vy_e=(self.y_e_est[count-self.delay]-self.y_e_est[count-self.delay-1])/dt

        input=[self.x_e_est[count-self.delay],vx_e,self.y_e_est[count-self.delay],vy_e,self.x_p_est[count-self.delay],self.y_p_est[count-self.delay]]#,self.delay

        #scaler = MinMaxScaler(feature_range=(0, 1))
        #scaler_input = scaler.fit(np.asarray(input).reshape(-1, 1))
        #input = scaler.transform(np.asarray(input).reshape(-1, 1))

        input = np.reshape( np.asarray(input), (1, 1,len(input)))
        input= torch.from_numpy(input).float()
        input.to(device)

        out, hidden = model(input,device)
        prob = nn.functional.softmax(out[-1], dim=0).data
        out = torch.max(prob, dim=0)[1].item()

        #print(out)

        plus=np.pi
        if out==0:
            psi =9*np.pi/8+plus#1#
        if out==1:
            psi =7*np.pi/8+plus#-1#
        if out==2:
            psi =15*np.pi/8+plus#-0.5#1#
        if out==3:
            psi =np.pi/8+plus#0.5#1#
        if out==4:
            psi =11*np.pi/8+plus#0#-1#
        if out==5:
            psi =13*np.pi/8+plus#-0.5#-1#
        if out==6:
            psi =5*np.pi/8+plus#0#-0.5#0#
        if out==7:
            psi =3*np.pi/8+plus#0#0.5#0#


        #PRIME
        #H=np.sign((psi-theta))#/dt
        H=(psi-theta)/dt
        #print(H)
        #theta=theta+(self.v_P/self.minR)*H#*dt

        #theta=(1/(self.v_P/self.minR))*psi
        #omega=(self.v_P/self.minR)*np.sin(psi) #maybe adjust psi
        theta=theta+(self.v_P/self.minR)*H*dt#omega*dt
        #theta=psi*(self.v_P/self.minR)
        #theta=psi
        x_p_step=self.x_p_est[count] + self.v_P*np.cos(theta)*dt
        y_p_step=self.y_p_est[count] + self.v_P*np.sin(theta)*dt

        #Prime
        #x_p_step=self.x_p_est[count] + (self.v_P)*np.cos(theta)*dt
        #y_p_step=self.y_p_est[count] + (self.v_P)*np.sin(theta)*dt


        #omega=self.v_P*np.sin(psi)/self.minR
        #arc=np.arctan2(self.y_p_est[count],self.x_p_est[count])

        #x_p_step=self.minR*np.cos(omega*dt+arc)-self.minR
        #y_p_step=self.minR*np.sin(omega*dt+arc)

        return x_p_step,y_p_step,theta

    #differential game: calculate trajectories
    def HCForwardTimeGlobal(self, figurecounter,two_Drones,predict_Kal,model,device):

        self.caught = 0
        theta_p = 0
        theta_p_delay=0
        theta_p_est=0
        dt = 0.01

        self.initialization()

        #build Kalman model
        predict_Kal.pos_init=np.array([self.x_e[0],0 ,self.y_e[0], 0])
        predict_Kal.v_E=self.v_E
        predict_Kal.build_Kalman()

        mu_x=[]
        mu_y=[]


        if False:
            for count,t in enumerate(self.time):

                #calculation of relative distance of pursuer and evader
                xRel = (self.x_e[count] - self.x_p[count])*np.cos(theta_p) + (self.y_e[count] - self.y_p[count])*np.sin(theta_p)
                yRel = -(self.x_e[count] - self.x_p[count])*np.sin(theta_p) + (self.y_e[count] - self.y_p[count])*np.cos(theta_p)
                #check whether catch has occured
                if xRel**2 + yRel**2 < self.captureL**2:
                    self.caught = 1
                    break

                #calculation of angular velocities
                if yRel == 0:
                    if xRel < 0:
                        phi = 1
                    else:
                        phi = 0
                else:
                    phi = np.sign(yRel)

                theta_p = theta_p + (self.v_P/self.minR)*phi*dt

                psi = theta_p + np.arctan2(yRel, xRel)

                self.theta_e = psi

                #calculation of next step of pursuer and evader without delay
                x_e_step=self.x_e[count] + self.v_E*np.cos(self.theta_e)*dt
                y_e_step=self.y_e[count] + self.v_E*np.sin(self.theta_e)*dt

                x_p_step=self.x_p[count] + self.v_P*np.cos(theta_p)*dt
                y_p_step=self.y_p[count] + self.v_P*np.sin(theta_p)*dt


                #delay x_p
                x_p_step,y_p_step,theta_p_delay=self.add_delay(theta_p_delay,dt, count)

                #save trajectories as array
                self.x_e.append(x_e_step)
                self.y_e.append(y_e_step)

                self.x_p.append(x_p_step)
                self.y_p.append(y_p_step)

        #if False:

        self.caught_est=0
        theta_p = 0
        #theta_p_delay=0
        theta_p_est=0
        dt = 0.01
        theta=0

        for count,t in enumerate(self.time):

            #calculation of relative distance of pursuer and evader
            xRel = (self.x_e_est[count] - self.x_p_est[count])*np.cos(theta_p) + (self.y_e_est[count] - self.y_p_est[count])*np.sin(theta_p)
            yRel = -(self.x_e_est[count] - self.x_p_est[count])*np.sin(theta_p) + (self.y_e_est[count] - self.y_p_est[count])*np.cos(theta_p)
            #check whether catch has occured
            if xRel**2 + yRel**2 < self.captureL**2:
                self.caught_est = 1
                break

            #calculation of angular velocities
            if yRel == 0:
                if xRel < 0:
                    phi = 1
                else:
                    phi = 0
            else:
                phi = np.sign(yRel)

            theta_p = theta_p + (self.v_P/self.minR)*phi*dt

            psi = theta_p + np.arctan2(yRel, xRel)

            self.theta_e = psi

            #calculation of next step of pursuer and evader without delay
            x_e_step=self.x_e_est[count] + self.v_E*np.cos(self.theta_e)*dt
            y_e_step=self.y_e_est[count] + self.v_E*np.sin(self.theta_e)*dt

            #x_p_step,y_p_step=self.newcontroller(count,model,device)

            #predict Kalman
            #depending on the delay -> estimation of estimation of estimation -> for loop
            if len(self.x_e_est)>self.delay: #and self.delay!=0
                x_p_step,y_p_step,theta_p_est=self.newcontroller(count,model,device,theta_p_est)#theta_p_est?
            #    self.execution_Kalman(predict_Kal,count,theta_p)#mu=
            #    x_p_step,y_p_step=self.newcontroller(count)
                #self.xe_predicted=np.concatenate((self.xe_predicted, mu[len(mu)-1,0]), axis=None)
                #self.ye_predicted=np.concatenate((self.ye_predicted, mu[len(mu)-1,2]), axis=None)
            else:
                self.xe_predicted.append(x_e_step)
                self.ye_predicted.append(y_e_step)
                x_p_step,y_p_step,theta_p_est=self.pursuers_trajectory_with_estimation(theta_p_est,dt,count)
                #self.xe_predicted.append(0)
                #self.ye_predicted.append(0)

            #print(len(self.xe_predicted))
            #print(len(self.ye_predicted))
            #print("after estimation final")
            #x_p_step,y_p_step,theta_p_est=self.pursuers_trajectory_with_estimation(theta_p_est,dt,count)
            #save trajectories as array
            self.x_e_est.append(x_e_step)
            self.y_e_est.append(y_e_step)

            self.x_p_est.append(x_p_step)
            self.y_p_est.append(y_p_step)

            #self.x_p_est.append(x_p_step_est)
            #self.y_p_est.append(y_p_step_est)

            #move the Drones in AirSim
            #two_Drones.move_drones(x_e_step,y_e_step,x_p_step,y_p_step)

        self.plotting(figurecounter, mu_x, mu_y)
        #self.msqe_prediction()
