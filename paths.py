import numpy as np
#030,013,016 = shit
def paths():
    files=[]
    #263
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/128/Trajectory/20080424121547.plt')
    #438
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/128/Trajectory/20080510013809.plt')
    #89
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/003/Trajectory/20081118012900.plt')
    #123
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/003/Trajectory/20081223185007.plt')
    #203
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/003/Trajectory/20090208020837.plt')
    #88
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/003/Trajectory/20090510222007.plt')
    #100
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/003/Trajectory/20090512195558.plt')
    #3675
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/003/Trajectory/20090523011253.plt')
    #21
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/000/Trajectory/20081029092138.plt')
    #12
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/000/Trajectory/20081116085532.plt')
    #21
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/000/Trajectory/20081123102153.plt')
    #3675
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/000/Trajectory/20090523011253.plt')
    #24428
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20080801023537.plt')
    #2716
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20081003152516.plt')
    #3462
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20090213101841.plt')
    #3940
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20090207081853.plt')
    #372
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20081231180207.plt')
    #6523
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20081231135628.plt')
    #4782
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20081213022402.plt')
    #362
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20081106225500.plt')
    #110
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20081015232937.plt')
    #1177
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20081003221227.plt')
    #6254
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20080617094444.plt')
    #882
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20080613211847.plt')
    #70
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20080328144824.plt')
    #55
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20071117170827.plt')
    #406
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20071026213100.plt')
    #3
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20070919121546.plt')
    #3216
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20070906204521.plt')
    #581
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20070804155303.plt')
    #288
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/011/Trajectory/20081008091709.plt')
    #91
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/011/Trajectory/20081027111004.plt')
    #1369
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/011/Trajectory/20081118062020.plt')
    #3
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/011/Trajectory/20081121110241.plt')
    #26
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/011/Trajectory/20081212100514.plt')
    #84
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/012/Trajectory/20081028014826.plt')
    #23224
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/012/Trajectory/20081116064735.plt')
    #2068
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/014/Trajectory/20081026173641.plt')
    #587
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/014/Trajectory/20081028101509.plt')
    #514
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/014/Trajectory/20081105102527.plt')
    #78
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/014/Trajectory/20081105124707.plt')
    #4
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/014/Trajectory/20081106003719.plt')
    #653
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/014/Trajectory/20081116034326.plt')
    #4
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/014/Trajectory/20081203100600.plt')
    #499
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/014/Trajectory/20081210100815.plt')
    #455
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/015/Trajectory/20081031123730.plt')
    #355
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/015/Trajectory/20081123150707.plt')
    #339
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/017/Trajectory/20081106072250.plt')
    #66
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/017/Trajectory/20081204031304.plt')
    #41
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/017/Trajectory/20081207065529.plt')
    #46
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/017/Trajectory/20081213073550.plt')
    #461
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/017/Trajectory/20081231112945.plt')
    #74
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/017/Trajectory/20090108133941.plt')
    #28
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/017/Trajectory/20090121121404.plt')
    #936
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/128/Trajectory/20080930034934.plt')
    #1187
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/128/Trajectory/20081120003216.plt')
    #447
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/128/Trajectory/20090222040348.plt')
    #1981
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/128/Trajectory/20090305213221.plt')
    #8248
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20080505011836.plt')
    #10166
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20080507015023.plt')
    #2255
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/010/Trajectory/20080917225654.plt')
    #929
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20120516184103.plt')
    #3284
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20120511102727.plt')
    #96
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20120511102358.plt')
    #307
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20120415162028.plt')
    #1022
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20120405211638.plt')
    #425
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20111224074358.plt')
    #229
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20111107210600.plt')
    #254
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20111106212707.plt')
    #210
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20101026235703.plt')
    #133
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20101020134128.plt')
    #103
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20090227043703.plt')
    #1239
    files.append('Trainingsdata/Geolife Trajectories 1.3/Data/163/Trajectory/20090221032640.plt')
    return files

def DiffgamePaths():
    files=[]
    files_pursuer=[]
    #263
    betaVal=0.2
    baseR=8
    captureL = betaVal*baseR
    delay_array=[0,10,50,100,110,120,121,122,123,124,125,126,127,128,129,130,140,150,160,180,200]
    theta_vec=np.arange(0,np.pi+(np.pi/8),(np.pi/8))
    rInit_vec=np.arange(2*captureL,11.5*captureL,captureL)

    for delay in delay_array:
        for rInit in rInit_vec:
            for theta in theta_vec:
                files.append("Input_one_step_RNN/RNNTrajectories/"+str(delay)+str(rInit)+str(theta)+".txt")
                files_pursuer.append("Input_one_step_RNN/Pursuer/"+str(delay)+"_"+str(rInit)+"_"+str(theta)+".txt")
    return files, files_pursuer
