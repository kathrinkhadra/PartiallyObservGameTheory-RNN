import torch
from torch import nn
import numpy as np
import dataextraction
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import modelRNN
from pairing import pair, depair
from collections import Counter
import matplotlib.pyplot as plt
MAX_LENGTH = 50

def preparation():
##Get DATA##
    x_e,y_e,x_p,y_p=dataextraction.getting_data()
    input_seq,target_seq,test_input, test_target=data_prepprocessing(x_e,y_e,x_p,y_p)
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
        y_vec_p=y_p[j]
        x_vec_p=x_p[j]
        #if len(input_seq)>=100000:
        #    break
        for i,x_val in enumerate(x_vec):
            if len(test_target)<=1000000 and i%2==1:
                if i+1<len(x_vec):
                    x_vel=(x_vec[i+1]-x_val)/dt
                    y_vel=(y_vec[i+1]-y_vec[i])/dt
                    test_input.append([x_val,x_vel,y_vec[i],y_vel,x_vec_p[i],y_vec_p[i]])
                    xRel=x_val-x_vec[i+1]
                    yRel=y_vec[i]-y_vec[i+1]
                    if np.absolute(yRel)<=np.absolute(xRel):
                        if np.sign(xRel)==-1:
                            test_target.append(0)#forward
                        else:
                            test_target.append(1)#backward
                    else:
                        if np.sign(yRel)==-1:
                            test_target.append(2)#right
                        else:
                            test_target.append(3)#left
            if len(input_seq)<=5000000 and i%2==0:
                if i+1<len(x_vec):
                    x_vel=(x_vec[i+1]-x_val)/dt
                    y_vel=(y_vec[i+1]-y_vec[i])/dt
                    input_seq.append([x_val,x_vel,y_vec[i],y_vel,x_vec_p[i],y_vec_p[i]])
                    xRel=x_val-x_vec[i+1]
                    yRel=y_vec[i]-y_vec[i+1]
                    if np.absolute(yRel)<=np.absolute(xRel):
                        if np.sign(xRel)==-1:
                            target_seq.append(0)#forward
                        else:
                            target_seq.append(1)#backward
                    else:
                        if np.sign(yRel)==-1:
                            target_seq.append(2)#right
                        else:
                            target_seq.append(3)#left

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
    scaler_input = scaler.fit(input_seq)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #target_seq = np.reshape(target_seq,(len(target_seq), 1))
    #scaler_target = scaler.fit(target_seq)

    input_seq = scaler.transform(input_seq)
    #add=min(target_seq)
    #target_seq=[x-add for x in target_seq]
    #target_seq = scaler.transform(target_seq)
    #print(target_seq.shape)
    #target_seq = np.reshape(target_seq,(len(target_seq)))
    #print(target_seq.shape)
    #print(input_seq)
    seq_input=[]
    for i,pairs in enumerate(input_seq):
        if i+50<len(input_seq):
            #print(pairs)
            seq_input.append(input_seq[i:i+50])
    input_seq=np.asarray(seq_input)
    print(input_seq.shape)
    #print(len(input_seq[0]))
    #print(len(input_seq[2]))

    #input_seq = np.reshape(input_seq, (input_seq.shape[0], 50,4))
    #print(input_seq)
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

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def build_model(device,input_seq,target_seq,test_input, test_target):

    # Instantiate the model with hyperparameters
    model = modelRNN.Model(input_size=4, output_size=4, hidden_dim=4, n_layers=2)#batch_size=200
    # We'll also set the model to the device that we defined earlier (default is CPU)
    model.to(device)

    # Define hyperparameters
    lr=0.01
    n_epochs = 240
    batches=100000
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
        for test in test_input[0:1000]:#slice it
            test=np.asarray(test)
            test = np.reshape( test, (1, 1,4))
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
        for i,target in enumerate(test_target[0:1000]):
            #print(target)
            #print(sequence[i])
            if target==sequence[i]:
                right=right+1
        test_loss.append(right/len(test_target[0:1000]))

        if epoch%10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))



    fig= plt.figure()
    plt.plot(range(1, n_epochs + 1),loss_value)
    plt.plot(range(1, n_epochs + 1),test_loss)
    fig.savefig('minibatchplots/Loss.png')
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
    path='models/test.pt'

    device,input_seq,target_seq,test_input, test_target=preparation()
    model=build_model(device,input_seq,target_seq,test_input, test_target)
    #test_input=np.asarray([39.78,50,143.56,60])
    #test_input = np.reshape(test_input, (1, 1,2))
    #save + load a model
    torch.save(model.state_dict(), path)
    model = modelRNN.Model(input_size=4, output_size=4, hidden_dim=4, n_layers=2)
    model.load_state_dict(torch.load(path))
    model.eval()

    sequence=[]
    #for i in range(0,10):
    #print(test_target)
    for test in test_input:
        test=np.asarray(test)
        test = np.reshape( test, (1, 1,4))
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
