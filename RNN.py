import torch
from torch import nn
import numpy as np
import dataextraction
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import modelRNN
from pairing import pair, depair
from collections import Counter
from random import choice
import matplotlib.pyplot as plt


def preparation():
##Get DATA##
    #print('Reading Data....')
    #x_e,y_e,x_p,y_p=dataextraction.getting_data()
    print('Loading Data....')
    #input_seq,target_seq,test_input, test_target=data_prepprocessing(x_e,y_e,x_p,y_p)
    input_seq=np.load('numpyarrays/one-step-input/input.npy')#'numpyarrays/input.npy')
    target_seq=np.load('numpyarrays/one-step-input/target.npy')#'numpyarrays/target.npy')
    test_input=np.load('numpyarrays/one-step-input/input_test.npy')#'numpyarrays/input_test.npy')
    test_target=np.load('numpyarrays/one-step-input/target_test.npy')#'numpyarrays/target_test.npy')
    #print(input_seq[:,:,-1])
    #sequence=[10,50,100,110,120,121,122,123,124,125,126,127,128,129,130,140,150,160,180,200]
    #delay = choice(sequence)
    #print(delay)
    input_seq = torch.from_numpy(input_seq).float()
    target_seq = torch.Tensor(target_seq).float()
    print('Finished Loading....')
    device=GPU()
    return device,input_seq,target_seq,test_input, test_target

def data_prepprocessing(x_e,y_e,x_p,y_p):
    ##DATA PREPROCESSING##
    # Creating lists that will hold our input and target sequences
    input_seq = []
    target_seq = []

    test_input=[]
    test_target=[]

    dt = 0.01
    for j,x_vec in enumerate(x_e):
        y_vec=y_e[j]
        x_vec_p=x_p[j]
        y_vec_p=y_p[j]
        #if len(input_seq)>=100000:
        #    break
        sequence=[10,50,100,110,120,121,122,123,124,125,126,127,128,129,130,140,150,160,180,200]
        for i,x_val in enumerate(x_vec):
            delay = choice(sequence)
            if len(test_target)<=1000000 and i%2==1:
                if i+delay<len(x_vec):
                    x_vel=(x_vec[i+delay]-x_val)/dt
                    y_vel=(y_vec[i+delay]-y_vec[i])/dt
                    test_input.append([x_val,x_vel,y_vec[i],y_vel,x_vec_p[i],y_vec_p[i],delay])
                    xRel=x_val-x_vec[i+delay]
                    yRel=y_vec[i]-y_vec[i+delay]
                    if np.absolute(yRel)<=np.absolute(xRel):
                        if np.sign(xRel)==-1:
                            if np.sign(yRel)==-1:
                                test_target.append(0)#forward1
                            else:
                                test_target.append(1)#forward2
                        else:
                            if np.sign(yRel)==-1:
                                test_target.append(2)#backward1
                            else:
                                test_target.append(3)#backward2
                    else:
                        if np.sign(yRel)==-1:
                            if np.sign(xRel)==-1:
                                test_target.append(4)#right1
                            else:
                                test_target.append(5)#right2
                        else:
                            if np.sign(xRel)==-1:
                                test_target.append(6)#left1
                            else:
                                test_target.append(7)#left2
                    #if np.absolute(yRel)<=np.absolute(xRel):
                    #    if np.sign(xRel)==-1:
                    #        test_target.append(0)#forward
                    #    else:
                    #        test_target.append(1)#backward
                    #else:
                    #    if np.sign(yRel)==-1:
                    #        test_target.append(2)#right
                    #    else:
                    #        test_target.append(3)#left
            if len(input_seq)<=7000000 and i%2==0:
                if i+delay<len(x_vec):
                    x_vel=(x_vec[i+delay]-x_val)/dt
                    y_vel=(y_vec[i+delay]-y_vec[i])/dt
                    input_seq.append([x_val,x_vel,y_vec[i],y_vel,x_vec_p[i],y_vec_p[i],delay])
                    xRel=x_val-x_vec[i+delay]
                    yRel=y_vec[i]-y_vec[i+delay]
                    if np.absolute(yRel)<=np.absolute(xRel):
                        if np.sign(xRel)==-1:
                            if np.sign(yRel)==-1:
                                target_seq.append(0)#forward1
                            else:
                                target_seq.append(1)#forward2
                        else:
                            if np.sign(yRel)==-1:
                                target_seq.append(2)#backward1
                            else:
                                target_seq.append(3)#backward2
                    else:
                        if np.sign(yRel)==-1:
                            if np.sign(xRel)==-1:
                                target_seq.append(4)#right1
                            else:
                                target_seq.append(5)#right2
                        else:
                            if np.sign(xRel)==-1:
                                target_seq.append(6)#left1
                            else:
                                target_seq.append(7)#left2

                    # Remove firsts character for target sequence
                    #values = (x_vec[i+1],y_vec[i+1])        # (2251799813685249, 2251799813685249)
                    #encoded = pair(*values, safe=False)
                    #target_seq.append(encoded)
                    #target_seq.append(pair(x_vec[i+1],y_vec[i+1]))
                    #depair(encoded)
    print(len(target_seq))
    print(len(test_target))
    #print(len(input_seq))
    #batch_size=len(input_seq)
    #input_seq= normalize(input_seq)#, axis=1, norm='l1'
    #target_seq = normalize(target_seq)#, axis=1, norm='l1'
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_input = scaler.fit(input_seq[:][:6])
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #target_seq = np.reshape(target_seq,(len(target_seq), 1))
    #scaler_target = scaler.fit(target_seq)

    input_seq[:][:6] = scaler.transform(input_seq[:][:6])
    #add=min(target_seq)
    #target_seq=[x-add for x in target_seq]
    #target_seq = scaler.transform(target_seq)
    #print(target_seq.shape)
    #target_seq = np.reshape(target_seq,(len(target_seq)))
    #print(target_seq.shape)

    input_seq = np.reshape(input_seq, (len(input_seq), 1,7))
    #print(input_seq[:,:,-1])
    np.save('numpyarrays/input.npy',input_seq)
    np.save('numpyarrays/input_test.npy',test_input)
    np.save('numpyarrays/target.npy',target_seq)
    np.save('numpyarrays/target_test.npy',test_target)

    input_seq = torch.from_numpy(input_seq).float()
    target_seq = torch.Tensor(target_seq).float()

    #print(target_seq.shape)
    return input_seq,target_seq,test_input, test_target

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

def build_model(device,input_seq,target_seq,test_input, test_target,epochs,layers,batch_size,path):

    # Instantiate the model with hyperparameters
    model = modelRNN.Model(input_size=6, output_size=8, hidden_dim=4, n_layers=layers)#batch_size=200
    # We'll also set the model to the device that we defined earlier (default is CPU)
    model.to(device)
    file=open(path+"/"+str(layers)+str(epochs)+str(batch_size)+".txt", "w+")
    # Define hyperparameters
    lr=0.01
    n_epochs = epochs#240
    batches=batch_size#50000
    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()#nn.MSELoss()#nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_value=[]
    test_loss=[]
    # Training Run
    for epoch in range(1, n_epochs + 1):

        #input_seq,target_seq = shuffle(input_seq,target_seq)

        permutation = torch.randperm(input_seq.size()[0])

        for i in range(0,input_seq.size()[0], batches):
            optimizer.zero_grad() # Clears existing gradients from previous epoch

            indices = permutation[i:i+batches]
            batch_x, batch_y = input_seq[indices,:], target_seq[indices]

            batch_x = batch_x.to(device)
            output, hidden = model(batch_x,device)
            #print(target_seq.view(-1,2).long())
            #print(output.view(-1).shape)
            #print(target_seq.view(-1).shape)
            loss = criterion(output, batch_y.long())#.view(-1).long()
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

        loss_value.append(loss.item())

        sequence=[]
        #print(test_input[0:1000])
        #for test in test_input:#slice it [0:1000]
        #    test=np.asarray(test)
        #    test = np.reshape( test, (1, 1,4))
        #    out,hidden=execute_model(test,device,model)

            #    test_input=test_input.detach().numpy()
                #print(test_input)
        #    prob = nn.functional.softmax(out[-1], dim=0).data
                # Taking the class with the highest probability score from the output
        #    out = torch.max(prob, dim=0)[1].item()

            #test_input=test_input+add
            #test_input=depair(test_input)
        #    sequence.append(out)

        #right=0
        #for i,target in enumerate(test_target[0:1000]):
            #print(target)
            #print(sequence[i])
        #    if target==sequence[i]:
        #        right=right+1
        #test_loss.append(right/len(test_target[0:1000]))

        if epoch%10 == 0:
            #print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            #print("Loss: {:.4f}".format(loss.item()))
            file.writelines(str(epoch)+":..."+str(loss.item())+"\n")


    file.close()
    fig= plt.figure()
    plt.plot(range(1, n_epochs + 1),loss_value)
    #plt.plot(range(1, n_epochs + 1),test_loss)
    fig.savefig(path+'/'+str(layers)+str(epochs)+str(batch_size)+'Loss.png')
    plt.close(fig)

    return model

def execute_model(test_input,device,model):
    input= torch.from_numpy(test_input).float()
    input.to(device)

    out, hidden = model(input,device)

    #prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    #highest_out = torch.max(prob, dim=0)[1].item()

    return out, hidden


if __name__ == "__main__":
    layers=[2]#,4,6]
    epochs=[600,800]
    batch_sizes=[100000]#,50000,25000,10000]
    device,input_seq,target_seq,test_input, test_target=preparation()
    for layer in layers:
        for epoch in epochs:
            for batch_size in batch_sizes:
                string=str(layer)+","+str(epoch)+","+str(batch_size)+":"
                print(string)
                pathmodel=str(layer)+str(epoch)+str(batch_size)+'.pt'
                path_data='..'
                print("Start Training....")
                model=build_model(device,input_seq,target_seq,test_input, test_target,epoch,layer,batch_size,path_data)
                #test_input=np.asarray([39.78,50,143.56,60])
                #test_input = np.reshape(test_input, (1, 1,2))
                #save + load a model
                torch.save(model.state_dict(), pathmodel)
                model = modelRNN.Model(input_size=6, output_size=8, hidden_dim=4, n_layers=layer)
                model.load_state_dict(torch.load(pathmodel))
                model.eval()

                sequence=[]
                #for i in range(0,10):
                #print(test_target)
                print("Start Testing....")
                for test in test_input:
                    test=np.asarray(test)
                    test = np.reshape( test, (1, 1,6))
                    out,hidden=execute_model(test,device,model)

                    #    test_input=test_input.detach().numpy()
                        #print(test_input)
                    prob = nn.functional.softmax(out[-1], dim=0).data
                        # Taking the class with the highest probability score from the output
                    out = torch.max(prob, dim=0)[1].item()

                    #test_input=test_input+add
                    #test_input=depair(test_input)
                    sequence.append(out)

                right=0
                for i,target in enumerate(test_target):
                    #print(target)
                    #print(sequence[i])
                    if target==sequence[i]:
                        right=right+1
                print(right)

                percentage=right/len(test_target)
                string="Test Success"+str(layer)+","+str(epoch)+","+str(batch_size)+":"
                print(string)
                print(percentage)
    #print(sequence)
    #print(test_input)
    #x=[]
    #y=[]
    #for s in sequence:
#        x.append(s[0])
#        y.append(s[1])
#    fig= plt.figure()
#    plt.plot(x,y)
#    #fig.savefig('testplots/'+str(index)+'.png')
#    plt.show()
#    plt.close(fig)
    #training(optimizer,device,input_seq,target_seq,model,criterion)
