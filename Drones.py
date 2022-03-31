import airsim

class Drones:
    
    def __init__(self, client):
        self.client=client 

    #connecting to AirSim
    def connecting(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True, "Drone1")
        self.client.enableApiControl(True, "Drone2")
        self.client.armDisarm(True, "Drone1")
        self.client.armDisarm(True, "Drone2")

    #drones taking off
    def flying(self):
        airsim.wait_key('Press any key to takeoff')
        self.client.takeoffAsync(vehicle_name="Drone1").join() #pursuer
        self.client.takeoffAsync(vehicle_name="Drone2").join() #evader
        
    #drones moving to position x,y
    def move_drones(self,x_e,y_e,x_p,y_p):

        #vpx=0.7
        #vpy=0.7
        #vex=0.7
        #vey=0.7

        #self.client.moveByVelocityZAsync(vpx, vpy,-6, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0),vehicle_name="Drone1").join()
        #self.client.moveByVelocityZAsync(vex, vey,-6, 1, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0),vehicle_name="Drone2").join()
        self.client.moveToPositionAsync(x_p,y_p, -10, 5,vehicle_name="Drone1").join()
        self.client.moveToPositionAsync(x_e,y_e, -10, 5,vehicle_name="Drone2").join()

    #drones state
    def get_state(self):
        state1 = self.client.getMultirotorState(vehicle_name="Drone1") #all info on drone in here
        s = pprint.pformat(state1)
        print("state: %s" % s)