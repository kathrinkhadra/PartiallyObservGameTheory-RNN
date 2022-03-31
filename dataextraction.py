import numpy as np
import matplotlib.pyplot as plt
import os
import paths


def getting_data():
    data=[]
    data_pur=[]

    #path = 'Trainingsdata/Geolife Trajectories 1.3/Data/017'
    file=[]
    files = []
    # r=root, d=directories, f = files
    #for r, d, f in os.walk(path):
    #    for file in f:
    #        if '.plt' in file:
    #            files.append(os.path.join(r, file))

    #print(len(files))
    #file=[files[14],files[81],files[99],files[114],files[147],files[159],files[210]]#,files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[],files[]

    #files = []
    #for k in file:
    #    files.append(k)
    #    print(k)
    #    print('\n')

    #files=paths.paths()
    files, files_pursuer=paths.DiffgamePaths()

    #print(files_pursuer)

    for index,file in enumerate(files):
        data.append(np.genfromtxt(file,delimiter=','))#,skip_header=6,usecols=np.arange(0,2)
        data_pur.append(np.genfromtxt(files_pursuer[index],delimiter=','))

    #print(data)

    x_axes_e=[]
    y_axes_e=[]
    x_axes_p=[]
    y_axes_p=[]
    for index,dat in enumerate(data):
        x=[]
        y=[]
        for line in dat:
            x.append(line[0])
            y.append(line[1])
        x_axes_e.append(x)
        y_axes_e.append(y)
        x=[]
        y=[]
        #print(data_pur[index])
        for line in data_pur[index]:
            x.append(line[0])
            y.append(line[1])
        x_axes_p.append(x)
        y_axes_p.append(y)


    #data_samples=0
    #for index,x in enumerate(x_axes):
    #    data_samples=data_samples+np.size(x)
    #    print(np.size(x))
    #    fig= plt.figure()
    #    plt.plot(x,y_axes[index])
        #plt.title(files[index])
    #    fig.savefig('testplots/'+str(index)+'.png')
    #    plt.show()
    #    plt.close(fig)


    #9558465
    #print(data_samples)
    #-.....
    #print(100000-data_samples)
    return x_axes_e,y_axes_e,x_axes_p,y_axes_p

#if __name__ == "__main__":
#    x_e,y_e,x_p,y_p=getting_data()
#    print(len(x_e))
#    print(len(y_e))
#    print(len(x_p))
#    print(len(y_p))
