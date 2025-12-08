import torch
import torch.nn as nn
import numpy as np
import math

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class STRF(nn.Module):
    def __init__(self, params):
        super(STRF, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.output_size = params['output_size'] # output layer size
        
        self.l1 = nn.Linear(self.input_size, self.output_size, bias=True) # linear layer (input-to-output)
        
        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.initialisation = params['initialisation']
        
        # parameter initialisation (could explore additional ones)
        # glorot initialisation: parameter values drawn from uniform distributions defined by network size
        if self.initialisation == 'glorot':
            for module in self.children():
                if isinstance(module, nn.Linear):
                    module.weight.data.uniform_(np.random.uniform(-1/np.sqrt(self.input_size + self.output_size), 
                                                                  1/np.sqrt(self.input_size + self.output_size)))
                    if module.bias is not None:
                        module.bias.data.uniform_(np.random.uniform(-1/np.sqrt(self.input_size + self.output_size), 
                                                                    1/np.sqrt(self.input_size + self.output_size)))
        
    def forward(self, x, stage):
        # forward pass of data through network
        # input x dimensions -> ()
        
        if self.dropout is True and stage == 'train':
            out = self.l1(self.input_dropout(x))
        else:
            out = self.l1(x)
        return out
    
    def forward_loop(self, inputs, target_data, stage):
        inputs = inputs.view(self.input_size, -1)
        inputs = torch.transpose(inputs, 0, 1).to(torch.float32)
        
        targets = target_data.transpose(0, 1).to(torch.float32)
        
        outputs = self.forward(inputs, stage)
        
        return outputs, targets
    
    def regul(self, lamb):
        regul_loss = lamb*(self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):
        model_info = {
            'w_l1': self.l1.weight, # weights - linear layer
            'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload linear layer
        self.l1 = nn.Linear(self.input_size, self.output_size, bias=True).to(device)
        self.l1.weight = nn.Parameter(info['w_l1'].to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(info['b_l1'].to(torch.float32).to(device))

class pop_STRF(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        '''Pennington and David params (bin_size = 10):
        self.n_F = 18
        self.n_filters_1 = 120
        self.filter_len_1 = 25
        self.n_out = 849'''
        
        self.n_F = params['n_F'] # number of frequency channels
        self.n_filters_1 = params['n_filters_1'] # number of filters in conv layer
        self.filter_len_1 = params['filter_len_1'] # length (in time bins) of filters in conv layer
        self.n_out = params['n_out'] # output (linear) layer size

        self.conv1 = torch.nn.Conv1d(self.n_F, self.n_filters_1, self.filter_len_1) # conv layer
        self.l1 = torch.nn.Linear(self.n_filters_1, self.n_out) # linear layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        self.resp_size = params['resp_size'] # response length (in time bins)

        #self.init_weight(self.l1.weight, "glorot_uniform")

    def forward(self, x, stage):
        # forward pass of data through network
        # input x dimensions -> (f, t) or (n, f, t) where n is number of stimuli

        x = self.conv1(x) # convolutional layer, output dimensions -> (filters, t) or (n, filters, t) with n as above
        
        if len(x.shape) == 2: # if no stimulus dimension, permute conv layer output to (t, filters)
            x = x.permute(1, 0)
        if len(x.shape) == 3: # if there is a stimulus dimension, permute conv layer output to (n, t, filters)
            x = x.permute(0, 2, 1)
            
        # linear layer, output dimensions -> (t, n_out) or (n, t, n_out) with n as above and usually n_out = neurons
        x = self.l1(x)

        return x
    
    def forward_loop(self, inputs, target_data, stage):
        # difference between stimulus and response size (may be due to zero-padding and tensorisation history having 
        # different lengths)
        stim_resp_diff = self.stim_size - self.resp_size

        # de-tensorise cochleagrams: only keep final element on history axis at each time point, then remove history 
        # dimension; tensorisation is useless for convolutional models as they take stimulus history into account through
        # filter length
        inputs = torch.squeeze(inputs[:, -1, :])

        # if input length is larger than the length of a stimulus (minibatch made up of more than one stimulus), reshape
        # input from dimensions (f, concatenated t) to (n, f, t), where n is the number of stimuli in the input minibatch
        # NOTE: models only work if minibatches include integer numbers of stimulus/response pairs
        if inputs.size(1) > self.stim_size:
            inputs = inputs.view(self.n_F, -1, self.stim_size) # reshape to (f, n, t)
            inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)

        # forward pass -> get response prediction output (dimensions -> (t, neurons) or (n, t, neurons))
        outputs = self.forward(inputs, stage)
        
        target_data = target_data.transpose(0, 1).to(torch.float32) # transpose target data to (t, neurons)

        # if target length is larger than the length of a response (minibatch made up of more than one responses), reshape
        # target from dimensions (concatenated t, neurons) to (n, t, neurons), where n is the number of responses in the 
        # target minibatch
        if target_data.size(0) > self.resp_size:
            targets = target_data.view(int(target_data.size(0)/self.resp_size), self.resp_size, -1)
            
            # if target length (in time) is larger than output length (may be due to mismatch in zero_padding bins and 
            # convolutional filter length), remove bins at the start of time axis corresponding to convolutional filter 
            # lengths while taking stimulus-response size difference into account
            if targets.size(1) > outputs.size(1):
                targets = targets[:, self.filter_len_1-1-stim_resp_diff:, :]
                
            # maybe must add option where output length is larger than target length, could be due to zero-padding being
            # being longer than convolutional filter length - would have to clip first few bins off output in that case
        
        # if target length is equal to response length but larger than output length, also adjust time axis accordingly as
        # above
        elif target_data.size(0) == self.resp_size and target_data.size(0) > outputs.size(0):
            targets = target_data[self.filter_len_1-1-stim_resp_diff:, :]
            
        # if target length is equal to output length, leave target unchanged
        elif target_data.size(0) == self.resp_size and target_data.size(0) == outputs.size(0):
            targets = target_data

        return outputs, targets
            
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.conv1.weight.norm(p=1) + self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):
        model_info = {
            'w_conv1': self.conv1.weight, # weights - convolutional layer
            'b_conv1': self.conv1.bias, # biases - convolutional layer
            'w_l1': self.l1.weight, # weights - linear layer
            'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload convolutional layer
        self.conv1 = torch.nn.Conv1d(self.n_F, self.n_filters_1, self.filter_len_1)
        self.conv1.weight = nn.Parameter(torch.tensor(info['w_conv1']).to(torch.float32).to(device))
        self.conv1.bias = nn.Parameter(torch.tensor(info['b_conv1']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = torch.nn.Linear(self.n_filters_1, self.n_out)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x):
        w_conv1 = self.conv1.weight.detach().cpu().numpy()
        b_conv1 = self.conv1.bias.detach().cpu().numpy()
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        
        if len(x.shape) == 2:
            x = np.expand_dims(x, 0)

        conv_out_t = np.zeros((np.size(x, 0), self.n_filters_1, self.stim_size+1-self.filter_len_1))
        for clip in range(np.size(x, 0)):
            for filt in range(self.n_filters_1):
                for t in range(self.filter_len_1-1, self.stim_size):
                    in_x = x[clip, :, t-self.filter_len_1+1:t+1]
                    conv_prod = np.sum(np.multiply(w_conv1[filt], in_x))
                    
                    conv_out_t[clip, filt, t-self.filter_len_1+1] = conv_prod + b_conv1[filt]
        
        o_trace = np.zeros((np.size(conv_out_t, 0), self.n_out, np.size(conv_out_t, -1)))
        
        for clip in range(np.size(x, 0)):
            x_in = conv_out_t[clip]
            o_t = np.dot(w_l1, x_in) + b_l1[:, None]
            
            o_trace[clip] = o_t
                
        traces = {'conv_out': conv_out_t, 'o': o_trace}
                
        return traces
    
    def np_forward_loop(self, inputs):
        inputs = torch.squeeze(inputs[:, -1, :])
        
        if inputs.size(1) > self.stim_size:
            inputs = inputs.view(self.n_F, -1, self.stim_size) # reshape to (f, n, t)
            inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)
        
        inputs = inputs.detach().cpu().numpy()
        
        traces = self.np_forward(inputs)

        return traces

############################################################################################################################# 

class pop_NRF(nn.Module):
    def __init__(self, params):
        super(pop_NRF, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        self.l1 = nn.Linear(self.input_size, self.hidden_size, bias=True) # 1st linear layer (input-to-hidden)
        self.l2 = nn.Linear(self.hidden_size, self.output_size, bias=True) # 2nd linear layer (hidden-to-output)
        
        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.initialisation = params['initialisation']
        
        # parameter initialisation
        # glorot initialisation: parameter values drawn from uniform distributions defined by network size
        if self.initialisation == 'glorot':
            for module in self.children():
                if isinstance(module, nn.Linear):
                    module.weight.data.uniform_(np.random.uniform(-1/np.sqrt(self.input_size + self.output_size), 
                                                                  1/np.sqrt(self.input_size + self.output_size)))
                    if module.bias is not None:
                        module.bias.data.uniform_(np.random.uniform(-1/np.sqrt(self.input_size + self.output_size), 
                                                                    1/np.sqrt(self.input_size + self.output_size)))
        
    def forward(self, x, stage):
        # forward pass of data through network
        # input x dimensions -> ()
        
        if self.dropout is True and stage == 'train': # use dropout only if it is on, and if the model is being trained
            out = self.l1(self.input_dropout(x)) # linear layer, output dimensions ->
            out = torch.sigmoid(out) # sigmoid nonlinearity
            
            out = self.l2(self.hidden_dropout(out)) # linear layer, output dimensions ->
            out = torch.sigmoid(out) # sigmoid nonlinearity
            
        else:
            out = self.l1(x) # linear layer, output dimensions ->
            out = torch.sigmoid(out) # sigmoid nonlinearity
            out = self.l2(out) # linear layer, output dimensions ->
            #out = torch.sigmoid(out) # sigmoid nonlinearity
            
        return out
    
    def forward_loop(self, inputs, target_data, stage):
        inputs = inputs.view(self.input_size, -1) # reshape input from () to ()
        inputs = torch.transpose(inputs, 0, 1).to(torch.float32) # transpose first two dimensions of input

        outputs = self.forward(inputs, stage) # forward pass -> get response prediction output

        targets = target_data.transpose(0, 1).to(torch.float32) # transpose target data to 

        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.l1.weight.norm(p=1) + self.l2.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):
        model_info = {
            'w_l1': self.l1.weight, # weights - 1st linear layer
            'b_l1': self.l1.bias, # biases - 1st linear layer
            'w_l2': self.l2.weight, # weights - 2nd linear layer
            'b_l2': self.l2.bias} # biases - 2nd linear layer
        
        return model_info
    
    def reload(self, info):
        # reload first linear layer
        self.l1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
        # reload second linear layer
        self.l2 = nn.Linear(self.hidden_size, self.output_size, bias=True)
        self.l2.weight = nn.Parameter(torch.tensor(info['w_l2']).to(torch.float32).to(device))
        self.l2.bias = nn.Parameter(torch.tensor(info['b_l2']).to(torch.float32).to(device))
        
    def np_forward(self, x):
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        
        w_l2 = self.l2.weight.detach().cpu().numpy()
        b_l2 = self.l2.bias.detach().cpu().numpy()

        h_pre_act = np.dot(w_l1, x) + b_l1[:, None]
        h = np.squeeze(1/(1 + np.exp(-h_pre_act)))
        
        o_pre_act = np.dot(w_l2, h) + b_l2[:, None]
        o = np.squeeze(1/(1 + np.exp(-o_pre_act)))
        
        traces = {'h_pre_act': h_pre_act, 'h': h, 'o_pre_act': o_pre_act, 'o': o}
        
        return traces
        
    def np_forward_loop(self, inputs):
        inputs = inputs.view(self.input_size, -1) # reshape input from () to ()
        inputs = inputs.detach().cpu().numpy()

        all_traces = self.np_forward(inputs) # forward pass -> get response prediction output
        
        return all_traces

#############################################################################################################################  
    
class pop_RNN(nn.Module):
    def __init__(self, params):
        super(pop_RNN, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        self.rnn = nn.RNN(self.input_size, self.hidden_size) # vanilla RNN layer (input-to-hidden)
        self.l1 = nn.Linear(self.hidden_size, self.output_size) # linear layer (hidden-to-output)
        
        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        self.initialisation = params['initialisation']
        
        # parameter initialisation
        # identity initialisation: hidden-to-hidden weights in vanilla RNN layer initialised to identity matrix 
        # (self-connections only); all other parameters with default PyTorch initialisation
        if self.initialisation == 'identity':
            self.rnn = nn.RNN(self.input_size, self.hidden_size)
            
            for name, param in self.rnn.named_parameters():
                if 'weight_hh' in name:
                    weight_hh = torch.eye(self.hidden_size, self.hidden_size) # identity matrix initialisation
                    param.data = weight_hh
                    
                if 'bias_hh' in name:
                    bias_hh = torch.zeros(self.hidden_size)
                    param.data = bias_hh
                    
                if 'weight_ih' in name:
                    weight_ih = torch.normal(torch.zeros(self.hidden_size, self.input_size), 
                                             0.001*torch.ones(self.hidden_size, self.input_size))
                    param.data = weight_ih
        
    def forward(self, x, h_0, stage):
        # forward pass of data through network
        # input x dimensions -> ()
        
        if self.dropout is True and stage == 'train':
            out, h_n = self.rnn(self.input_dropout(x), h_0)
            out = torch.sigmoid(self.l1(self.hidden_dropout(out)))
        else:
            '''out, h_n = self.rnn(x, h_0)
            out = torch.sigmoid(self.l1(out))'''
            
            out, h_n = self.rnn(x, h_0)
            out = self.l1(out)
            
        return out, h_n
    
    def forward_loop(self, inputs, target_data, stage):
        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device) # initial hidden state, needed as input to the RNN
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device) 
            input_t = inputs[:, :, t].view(-1, self.input_size)
            
            # forward
            output, h_n = self.forward(input_t, h_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
        
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.rnn.weight_ih_l0.norm(p=1) + self.rnn.weight_hh_l0.norm(p=1) + self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):
        model_info = {
            'w_rnn_ih': self.rnn.weight_ih_l0, # weights - vanilla RNN layer, input-to-hidden
            'b_rnn_ih': self.rnn.bias_ih_l0, # biases - vanilla RNN layer, input-to-hidden
            'w_rnn_hh': self.rnn.weight_hh_l0, # weights - vanilla RNN layer, hidden-to-hidden
            'b_rnn_hh': self.rnn.bias_hh_l0, # biases - vanilla RNN layer, hidden-to-hidden
            'w_l1': self.l1.weight, # weights - linear layer
            'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload vanilla RNN layer
        self.rnn = nn.RNN(self.input_size, self.hidden_size)
        self.rnn.weight_ih_l0 = nn.Parameter(torch.tensor(info['w_rnn_ih']).to(torch.float32).to(device))
        self.rnn.bias_ih_l0 = nn.Parameter(torch.tensor(info['b_rnn_ih']).to(torch.float32).to(device))
        self.rnn.weight_hh_l0 = nn.Parameter(torch.tensor(info['w_rnn_hh']).to(torch.float32).to(device))
        self.rnn.bias_hh_l0 = nn.Parameter(torch.tensor(info['b_rnn_hh']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
    
    def np_forward(self, x, h_t):
        w_rnn_ih = self.rnn.weight_ih_l0.detach().cpu().numpy()
        b_rnn_ih = self.rnn.bias_ih_l0.detach().cpu().numpy()
        w_rnn_hh = self.rnn.weight_hh_l0.detach().cpu().numpy()
        b_rnn_hh = self.rnn.bias_hh_l0.detach().cpu().numpy()
        
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()

        h_t_pre_act = np.dot(w_rnn_ih, x) + b_rnn_ih[:, None] + np.dot(w_rnn_hh, h_t) + b_rnn_hh[:, None]
        h_t = np.tanh(h_t_pre_act)
        
        o_t_pre_act = np.dot(w_l1, h_t) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
        
        return o_t, o_t_pre_act, h_t, h_t_pre_act    
    
    def np_forward_loop(self, inputs):
        h = np.zeros((self.hidden_size, 1)) # initial hidden state, needed as input to the RNN
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1)) 

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, h_pre_act = self.np_forward(input_t, h)

            if t == 0:
                h_pre_act_trace = np.array(h_pre_act)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                h_pre_act_trace = np.append(h_pre_act_trace, h_pre_act, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'h_pre_act': h_pre_act_trace, 'h': h_trace, 'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces

#############################################################################################################################
    
class pop_GRU(nn.Module):
    def __init__(self, params):
        super(pop_GRU, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        self.gru = nn.GRU(self.input_size, self.hidden_size) # GRU layer (input-to-hidden)
        self.l1 = nn.Linear(self.hidden_size, self.output_size) # linear layer (hidden-to-output)
        
        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, stage):
        # forward pass of data through network
        # input x dimensions -> ()
        
        if self.dropout is True and stage == 'train':
            out, h_n = self.gru(self.input_dropout(x), h_0)
            out = torch.sigmoid(self.l1(self.hidden_dropout(out)))
        else:
            '''out, h_n = self.gru(x, h_0)
            out = torch.sigmoid(self.l1(out))'''
            
            out, h_n = self.gru(x, h_0)
            out = self.l1(out)
                
        return out, h_n
    
    def forward_loop(self, inputs, target_data, stage):
        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device) # initial hidden state, needed as input to the RNN

        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
            
            input_t = inputs[:, :, t].view(-1, self.input_size)

            # forward
            output, h_n = self.forward(input_t, h_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
        
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.gru.weight_ih_l0.norm(p=1) + self.gru.weight_hh_l0.norm(p=1) + self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):
        model_info = {
             'w_gru_ih': self.gru.weight_ih_l0, # weights - GRU layer, input-to-hidden
             'b_gru_ih': self.gru.bias_ih_l0, # biases - GRU layer, input-to-hidden
             'w_gru_hh': self.gru.weight_hh_l0, # weights, GRU layer, hidden-to-hidden
             'b_gru_hh': self.gru.bias_hh_l0, # biases, GRU layer, hidden-to-hidden
             'w_l1': self.l1.weight, # weights - linear layer
             'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload GRU layer
        self.gru = nn.GRU(self.input_size, self.hidden_size)
        self.gru.weight_ih_l0 = nn.Parameter(torch.tensor(info['w_gru_ih']).to(torch.float32).to(device))
        self.gru.bias_ih_l0 = nn.Parameter(torch.tensor(info['b_gru_ih']).to(torch.float32).to(device))
        self.gru.weight_hh_l0 = nn.Parameter(torch.tensor(info['w_gru_hh']).to(torch.float32).to(device))
        self.gru.bias_hh_l0 = nn.Parameter(torch.tensor(info['b_gru_hh']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x, h_t):
        w_gru_ih = self.gru.weight_ih_l0.detach().cpu().numpy()
        b_gru_ih = self.gru.bias_ih_l0.detach().cpu().numpy()
        w_gru_hh = self.gru.weight_hh_l0.detach().cpu().numpy()
        b_gru_hh = self.gru.bias_hh_l0.detach().cpu().numpy()
        
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        
        r_t_pre_act = np.dot(w_gru_ih[:self.hidden_size, :], x) + b_gru_ih[:self.hidden_size][:, None] + \
            np.dot(w_gru_hh[:self.hidden_size, :], h_t) + b_gru_hh[:self.hidden_size][:, None]
        r_t = 1/(1 + np.exp(-r_t_pre_act))
        
        z_t_pre_act = np.dot(w_gru_ih[self.hidden_size:2*self.hidden_size, :], x) + \
            b_gru_ih[self.hidden_size:2*self.hidden_size][:, None] + \
            np.dot(w_gru_hh[self.hidden_size:2*self.hidden_size, :], h_t) + \
            b_gru_hh[self.hidden_size:2*self.hidden_size][:, None]
        z_t = 1/(1 + np.exp(-z_t_pre_act))
        
        n_t_pre_act = np.dot(w_gru_ih[2*self.hidden_size:, :], x) + b_gru_ih[2*self.hidden_size:][:, None] + \
            np.multiply(r_t, np.dot(w_gru_hh[2*self.hidden_size:, :], h_t) + b_gru_hh[2*self.hidden_size:][:, None])
        n_t = np.tanh(n_t_pre_act)
        
        h_t = np.multiply((1 - z_t), n_t) + np.multiply(z_t, h_t)
        
        o_t_pre_act = np.dot(w_l1, h_t) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
        
        return o_t, o_t_pre_act, h_t, n_t, n_t_pre_act, z_t, z_t_pre_act, r_t, r_t_pre_act
        
    def np_forward_loop(self, inputs):
        h = np.zeros((self.hidden_size, 1)) # initial hidden state, needed as input to the RNN
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1)) 

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, n, n_pre_act, z, z_pre_act, r, r_pre_act = self.np_forward(input_t, h)

            if t == 0:
                r_pre_act_trace = np.array(r_pre_act)
                r_trace = np.array(r)
                z_pre_act_trace = np.array(z_pre_act)
                z_trace = np.array(z)
                n_pre_act_trace = np.array(n_pre_act)
                n_trace = np.array(n)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                r_pre_act_trace = np.append(r_pre_act_trace, r_pre_act, axis=1)
                r_trace = np.append(r_trace, r, axis=1)
                z_pre_act_trace = np.append(z_pre_act_trace, z_pre_act, axis=1)
                z_trace = np.append(z_trace, z, axis=1)
                n_pre_act_trace = np.append(n_pre_act_trace, n_pre_act, axis=1)
                n_trace = np.append(n_trace, n, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'r_pre_act': r_pre_act_trace, 'r': r_trace, 'z_pre_act': z_pre_act_trace, 'z': z_trace, 
                  'n_pre_act': n_pre_act_trace, 'n': n_trace, 'h': h_trace, 'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces
        
#############################################################################################################################
        
class pop_LSTM(nn.Module):
    def __init__(self, params):
        super(pop_LSTM, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size) # LSTM layer (input-to-hidden)
        self.l1 = nn.Linear(self.hidden_size, self.output_size) # linear layer (hidden-to-output)
        
        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, c_0, stage):
        # forward pass of data through network
        # input x dimensions -> ()
        
        if self.dropout is True and stage == 'train':
            out, (h_n, c_n) = self.lstm(self.input_dropout(x), (h_0, c_0))
            out = torch.sigmoid(self.l1(self.hidden_dropout(out)))
        else:
            '''out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
            out = torch.sigmoid(self.l1(out))'''
            
            out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
            out = self.l1(out)
        
        return out, h_n, c_n
    
    def forward_loop(self, inputs, target_data, stage):
        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device)
        c_n = torch.zeros((1, self.hidden_size), device=device)
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
                c_n = torch.zeros((1, self.hidden_size), device=device)

            input_t = inputs[:, :, t].view(-1, self.input_size)
            
            # forward
            output, h_n, c_n = self.forward(input_t, h_n, c_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
        
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.lstm.weight_ih_l0.norm(p=1) + self.lstm.weight_hh_l0.norm(p=1) + self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):
        model_info = {
            'w_lstm_ih': self.lstm.weight_ih_l0, # weights - LSTM layer, input-to-hidden
            'b_lstm_ih': self.lstm.bias_ih_l0, # biases - LSTM layer, input-to-hidden
            'w_lstm_hh': self.lstm.weight_hh_l0, # weights - LSTM layer, hidden-to-hidden
            'b_lstm_hh': self.lstm.bias_hh_l0, # biases - LSTM layer, hidden-to-hidden
            'w_l1': self.l1.weight, # weights - linear layer
            'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.lstm.weight_ih_l0 = nn.Parameter(torch.tensor(info['w_lstm_ih']).to(torch.float32).to(device))
        self.lstm.bias_ih_l0 = nn.Parameter(torch.tensor(info['b_lstm_ih']).to(torch.float32).to(device))
        self.lstm.weight_hh_l0 = nn.Parameter(torch.tensor(info['w_lstm_hh']).to(torch.float32).to(device))
        self.lstm.bias_hh_l0 = nn.Parameter(torch.tensor(info['b_lstm_hh']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
    
    def np_forward(self, x, h_t, c_t):
        w_lstm_ih = self.lstm.weight_ih_l0.detach().cpu().numpy()
        b_lstm_ih = self.lstm.bias_ih_l0.detach().cpu().numpy()
        w_lstm_hh = self.lstm.weight_hh_l0.detach().cpu().numpy()
        b_lstm_hh = self.lstm.bias_hh_l0.detach().cpu().numpy()
        
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        
        i_t_pre_act = np.dot(w_lstm_ih[:self.hidden_size, :], x) + b_lstm_ih[:self.hidden_size][:, None] + \
            np.dot(w_lstm_hh[:self.hidden_size, :], h_t) + b_lstm_hh[:self.hidden_size][:, None]
        i_t = 1/(1 + np.exp(-i_t_pre_act))
        
        f_t_pre_act = np.dot(w_lstm_ih[self.hidden_size:2*self.hidden_size, :], x) + \
            b_lstm_ih[self.hidden_size:2*self.hidden_size][:, None] + \
            np.dot(w_lstm_hh[self.hidden_size:2*self.hidden_size, :], h_t) + \
            b_lstm_hh[self.hidden_size:2*self.hidden_size][:, None]
        f_t = 1/(1 + np.exp(-f_t_pre_act))
        
        out_t_pre_act = np.dot(w_lstm_ih[3*self.hidden_size:, :], x) + b_lstm_ih[3*self.hidden_size:][:, None] + \
            np.dot(w_lstm_hh[3*self.hidden_size:, :], h_t) + b_lstm_hh[3*self.hidden_size:][:, None]
        out_t = 1/(1 + np.exp(-out_t_pre_act))
        
        g_t_pre_act = np.dot(w_lstm_ih[2*self.hidden_size:3*self.hidden_size, :], x) + \
            b_lstm_ih[2*self.hidden_size:3*self.hidden_size][:, None] + \
            np.dot(w_lstm_hh[2*self.hidden_size:3*self.hidden_size, :], h_t) + \
            b_lstm_hh[2*self.hidden_size:3*self.hidden_size][:, None]
        g_t = np.tanh(g_t_pre_act)
        
        c_t = np.multiply(f_t, c_t) + np.multiply(i_t, g_t)
        
        h_t = np.multiply(out_t, np.tanh(c_t))
        
        o_t_pre_act = np.dot(w_l1, h_t) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
        
        return o_t, o_t_pre_act, h_t, c_t, out_t, out_t_pre_act, g_t, g_t_pre_act, f_t, f_t_pre_act, i_t, i_t_pre_act    
    
    def np_forward_loop(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, c, out, out_pre_act, g, g_pre_act, f, f_pre_act, i, i_pre_act = self.np_forward(input_t, h, c)

            if t == 0:
                i_pre_act_trace = np.array(i_pre_act)
                i_trace = np.array(i)
                f_pre_act_trace = np.array(f_pre_act)
                f_trace = np.array(f)
                g_pre_act_trace = np.array(g_pre_act)
                g_trace = np.array(g)
                out_pre_act_trace = np.array(out_pre_act)
                out_trace = np.array(out)
                c_trace = np.array(c)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                i_pre_act_trace = np.append(i_pre_act_trace, i_pre_act, axis=1)
                i_trace = np.append(i_trace, i, axis=1)
                f_pre_act_trace = np.append(f_pre_act_trace, f_pre_act, axis=1)
                f_trace = np.append(f_trace, f, axis=1)
                g_pre_act_trace = np.append(g_pre_act_trace, g_pre_act, axis=1)
                g_trace = np.append(g_trace, g, axis=1)
                out_pre_act_trace = np.append(out_pre_act_trace, out_pre_act, axis=1)
                out_trace = np.append(out_trace, out, axis=1)
                c_trace = np.append(c_trace, c, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'i_pre_act': i_pre_act_trace, 'i': i_trace, 'f_pre_act': f_pre_act_trace, 'f': f_trace,
                  'g_pre_act': g_pre_act_trace, 'g': g_trace, 'out_pre_act': out_pre_act_trace, 'out': out_trace,
                  'c': c_trace, 'h': h_trace, 'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces
        
#############################################################################################################################

class MGU1_layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MGU1_layer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.f_gate_w_x = nn.Linear(input_size, hidden_size)
        self.f_gate_w_h = nn.Linear(hidden_size, hidden_size)
        self.input_w_x = nn.Linear(input_size, hidden_size)
        self.input_w_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, h_0):
        f_gate = torch.sigmoid(self.f_gate_w_x(x) + self.f_gate_w_h(h_0))
        candidate_h = torch.tanh(self.input_w_x(x) + self.input_w_h(torch.multiply(f_gate, h_0)))
        
        h_t = torch.multiply((1 - f_gate), h_0) + torch.multiply(f_gate, candidate_h)
        out = h_t
        
        return out, h_t
    
class MGU2_layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MGU2_layer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.f_gate_w_h = nn.Linear(hidden_size, hidden_size)
        self.input_w_x = nn.Linear(input_size, hidden_size)
        self.input_w_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, h_0):
        f_gate = torch.sigmoid(self.f_gate_w_h(h_0))
        candidate_h = torch.tanh(self.input_w_x(x) + self.input_w_h(torch.multiply(f_gate, h_0)))
        
        h_t = torch.multiply((1 - f_gate), h_0) + torch.multiply(f_gate, candidate_h)
        out = h_t
        
        return out, h_t
    
class MGU3_layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MGU3_layer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.f_gate_w_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.input_w_x = nn.Linear(input_size, hidden_size)
        self.input_w_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, h_0):
        f_gate = torch.sigmoid(self.f_gate_w_h(h_0))
        candidate_h = torch.tanh(self.input_w_x(x) + self.input_w_h(torch.multiply(f_gate, h_0)))
        
        h_t = torch.multiply((1 - f_gate), h_0) + torch.multiply(f_gate, candidate_h)
        out = h_t
        
        return out, h_t

class pop_MGU(nn.Module):
    def __init__(self, params):
        super(pop_MGU, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        # MGU layer (input-to-hidden); depending on the type chosen, the form of the layer changes
        self.MGU_type = 1
        self.mgu = globals()['MGU' + str(self.MGU_type) + '_layer'](self.input_size, self.hidden_size)
        self.l1 = nn.Linear(self.hidden_size, self.output_size) # linear layer (hidden-to-output)
        
        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, stage):
        # forward pass of data through network
        # input x dimensions -> ()
        
        if self.dropout is True and stage == 'train':
            out, h_n = self.mgu(self.input_dropout((x), h_0)) # CHECK THIS
            out = torch.sigmoid(self.l1(self.hidden_dropout(out)))
        else:
            out, h_n = self.mgu(x, h_0)
            out = torch.sigmoid(self.l1(out))
            
        return out, h_n
    
    def forward_loop(self, inputs, target_data, stage):
        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device)
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
        
            input_t = inputs[:, :, t].view(-1, self.input_size)
            
            # forward
            output, h_n = self.forward(input_t, h_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
        
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.mgu.f_gate_w_h.weight.norm(p=1) + self.mgu.input_w_x.weight.norm(p=1) + \
                           self.mgu.input_w_h.weight.norm(p=1) + self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):
        if self.MGU_type == 1:
            model_info = {
                 'w_mgu_x_f': self.mgu.f_gate_w_x.weight, # weights - MGU layer, input-to-f gate
                 'b_mgu_x_f': self.mgu.f_gate_w_x.bias, # biases - MGU layer, input-to-f gate
                 'b_mgu_h_f': self.mgu.f_gate_w_h.bias} # biases - MGU layer, hidden-to-f gate
        
        elif self.MGU_type == 2:
            model_info = {
                 'b_mgu_h_f': self.mgu.f_gate_w_h.bias} # biases - MGU layer, hidden-to-f gate
        
        elif self.MGU_type == 3:
            model_info = {}
        
        model_info['w_mgu_h_f'] = self.mgu.f_gate_w_h.weight # weights - MGU layer, hidden-to-f gate
        model_info['w_mgu_x_in'] = self.mgu.input_w_x.weight # weights - MGU layer, input-to-hidden
        model_info['b_mgu_x_in'] = self.mgu.input_w_x.bias # biases - MGU layer, input-to-hidden
        model_info['w_mgu_h_in'] = self.mgu.input_w_h.weight # weights - MGU layer, hidden-to-hidden
        model_info['b_mgu_h_in'] = self.mgu.input_w_h.bias # biases - MGU layer, hidden-to-hidden
        
        model_info['w_l1'] = self.l1.weight # weights - linear layer
        model_info['b_l1'] = self.l1.bias # biases - linear layer
        
        return model_info
        
    def reload(self, info):
        # reload MGU layer
        if self.MGU_type == 1:
            self.mgu = MGU1_layer(self.input_size, self.hidden_size)
            self.mgu.f_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_mgu_x_f']).to(torch.float32).to(device))
            self.mgu.f_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_mgu_x_f']).to(torch.float32).to(device))
            self.mgu.f_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_mgu_h_f']).to(torch.float32).to(device))
        
        elif self.MGU_type == 2:
            self.mgu = MGU2_layer(self.input_size, self.hidden_size)
            self.mgu.f_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_mgu_h_f']).to(torch.float32).to(device))
        
        elif self.MGU_type == 3:
            self.mgu = MGU3_layer(self.input_size, self.hidden_size)
        
        self.mgu.f_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_mgu_h_f']).to(torch.float32).to(device))
        self.mgu.input_w_x.weight = nn.Parameter(torch.tensor(info['w_mgu_x_in']).to(torch.float32).to(device))
        self.mgu.input_w_x.bias = nn.Parameter(torch.tensor(info['b_mgu_x_in']).to(torch.float32).to(device))
        self.mgu.input_w_h.weight = nn.Parameter(torch.tensor(info['w_mgu_h_in']).to(torch.float32).to(device))
        self.mgu.input_w_h.bias = nn.Parameter(torch.tensor(info['b_mgu_h_in']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x, h_t):
        if self.MGU_type == 1:
            w_mgu_x_f = self.mgu.f_gate_w_x.weight.detach().cpu().numpy()
            b_mgu_x_f = self.mgu.f_gate_w_x.bias.detach().cpu().numpy()
            b_mgu_h_f = self.mgu.f_gate_w_h.bias.detach().cpu().numpy()
        
        elif self.MGU_type == 2:
            b_mgu_h_f = self.mgu.f_gate_w_h.bias.detach().cpu().numpy()
        
        elif self.MGU_type == 3:
            dummy = 0
        
        w_mgu_h_f = self.mgu.f_gate_w_h.weight.detach().cpu().numpy()
        w_mgu_x_in = self.mgu.input_w_x.weight.detach().cpu().numpy()
        b_mgu_x_in = self.mgu.input_w_x.bias.detach().cpu().numpy()
        w_mgu_h_in = self.mgu.input_w_h.weight.detach().cpu().numpy()
        b_mgu_h_in = self.mgu.input_w_h.bias.detach().cpu().numpy()
        
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        
        if self.MGU_type == 1:
            f_t_pre_act = np.dot(w_mgu_x_f, x) + b_mgu_x_f[:, None] + np.dot(w_mgu_h_f, h_t) + b_mgu_h_f[:, None]
            
        elif self.MGU_type == 2:
            f_t_pre_act = np.dot(w_mgu_h_f, h_t) + b_mgu_h_f[:, None]
            
        elif self.MGU_type == 3:
            f_t_pre_act = np.dot(w_mgu_h_f, h_t)
            
        f_t = 1/(1 + np.exp(-f_t_pre_act))
        
        cand_h_pre_act_t = np.dot(w_mgu_x_in, x) + b_mgu_x_in[:, None] + np.dot(w_mgu_h_in, np.multiply(f_t, h_t)) + \
            b_mgu_h_in[:, None]
        cand_h_t = np.tanh(cand_h_pre_act_t)
        
        h_t = np.multiply((1 - f_t), h_t) + np.multiply(f_t, cand_h_t)
        
        o_t_pre_act = np.dot(w_l1, h_t) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
        
        return o_t, o_t_pre_act, h_t, cand_h_t, cand_h_pre_act_t, f_t, f_t_pre_act
    
    def np_forward_loop(self, inputs):
        h = np.zeros((self.hidden_size, 1)) # initial hidden state, needed as input to the RNN
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1)) 

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, cand_h, cand_h_pre_act, f, f_pre_act = self.np_forward(input_t, h)

            if t == 0:
                f_pre_act_trace = np.array(f_pre_act)
                f_trace = np.array(f)
                cand_h_pre_act_trace = np.array(cand_h_pre_act)
                cand_h_trace = np.array(cand_h)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                f_pre_act_trace = np.append(f_pre_act_trace, f_pre_act, axis=1)
                f_trace = np.append(f_trace, f, axis=1)
                cand_h_pre_act_trace = np.append(cand_h_pre_act_trace, cand_h, axis=1)
                cand_h_trace = np.append(cand_h_trace, cand_h, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'f_pre_act': f_pre_act_trace, 'f': f_trace, 'cand_h': cand_h_trace, 'h': h_trace, 
                  'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces
    
#############################################################################################################################

class subLSTM_layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(subLSTM_layer, self).__init__()
        
        self.f_gate_w_x = nn.Linear(input_size, hidden_size)
        self.f_gate_w_h = nn.Linear(hidden_size, hidden_size)
        self.i_gate_w_x = nn.Linear(input_size, hidden_size)
        self.i_gate_w_h = nn.Linear(hidden_size, hidden_size)
        self.o_gate_w_x = nn.Linear(input_size, hidden_size)
        self.o_gate_w_h = nn.Linear(hidden_size, hidden_size)
        
        self.input_w_x = nn.Linear(input_size, hidden_size)
        self.input_w_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, h_0, c_0):
        f_gate = torch.sigmoid(self.f_gate_w_x(x) + self.f_gate_w_h(h_0))
        i_gate = torch.sigmoid(self.i_gate_w_x(x) + self.i_gate_w_h(h_0))
        o_gate = torch.sigmoid(self.o_gate_w_x(x) + self.o_gate_w_h(h_0))
        
        candidate_c = torch.sigmoid(self.input_w_x(x) + self.input_w_h(h_0))
        
        c_t = torch.multiply(f_gate, c_0) + candidate_c - i_gate
        
        h_t = torch.sigmoid(c_t) - o_gate
        out = h_t
        
        return out, h_t, c_t

class pop_subLSTM(nn.Module):
    def __init__(self, params):
        super(pop_subLSTM, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        # subLSTM layer (input-to-hidden)
        self.sub_lstm = subLSTM_layer(self.input_size, self.hidden_size)
        self.l1 = nn.Linear(self.hidden_size, self.output_size) # linear layer (hidden-to-output)

        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, c_0, stage):
        # forward pass of data through network
        # input x dimensions -> ()
        
        if self.dropout is True and stage == 'train':
            out, h_n, c_n = self.sub_lstm(self.input_dropout(x), h_0, c_0)
            out = torch.sigmoid(self.l1(self.hidden_dropout(out)))
        else:
            '''out, h_n, c_n = self.sub_lstm(x, h_0, c_0)
            out = torch.sigmoid(self.l1(out))'''
            
            out, h_n, c_n = self.sub_lstm(x, h_0, c_0)
            out = self.l1(out)
            
        return out, h_n, c_n
    
    def forward_loop(self, inputs, target_data, stage):
        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device)
        c_n = torch.zeros((1, self.hidden_size), device=device)
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
                c_n = torch.zeros((1, self.hidden_size), device=device)
        
            input_t = inputs[:, :, t].view(-1, self.input_size)
            
            # forward
            output, h_n, c_n = self.forward(input_t, h_n, c_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
        
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.sub_lstm.f_gate_w_x.weight.norm(p=1) + self.sub_lstm.f_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.i_gate_w_x.weight.norm(p=1) + self.sub_lstm.i_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.o_gate_w_x.weight.norm(p=1) + self.sub_lstm.o_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.input_w_x.weight.norm(p=1) + self.sub_lstm.input_w_h.weight.norm(p=1) + \
                           self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):        
        model_info = {
             'w_sublstm_x_f': self.sub_lstm.f_gate_w_x.weight, # weights - subLSTM layer, input-to-f gate
             'b_sublstm_x_f': self.sub_lstm.f_gate_w_x.bias, # biases - subLSTM layer, input-to-f gate
             'w_sublstm_h_f': self.sub_lstm.f_gate_w_h.weight, # weights - subLSTM layer, hidden-to-f gate
             'b_sublstm_h_f': self.sub_lstm.f_gate_w_h.bias, # biases - subLSTM layer, hidden-to-f gate
             'w_sublstm_x_i': self.sub_lstm.i_gate_w_x.weight, # weights - subLSTM layer, input-to-i gate
             'b_sublstm_x_i': self.sub_lstm.i_gate_w_x.bias, # biases - subLSTM layer, input-to-i gate
             'w_sublstm_h_i': self.sub_lstm.i_gate_w_h.weight, # weights - subLSTM layer, hidden-to-i gate
             'b_sublstm_h_i': self.sub_lstm.i_gate_w_h.bias, # biases - subLSTM layer, hidden-to-i gate
             'w_sublstm_x_o': self.sub_lstm.o_gate_w_x.weight, # weights - subLSTM layer, input-to-o gate
             'b_sublstm_x_o': self.sub_lstm.o_gate_w_x.bias, # biases - subLSTM layer, input-to-o gate
             'w_sublstm_h_o': self.sub_lstm.o_gate_w_h.weight, # weights - subLSTM layer, hidden-to-o gate
             'b_sublstm_h_o': self.sub_lstm.o_gate_w_h.bias, # biases - subLSTM layer, hidden-to-o gate
             'w_sublstm_x_in': self.sub_lstm.input_w_x.weight, # weights - subLSTM layer, input-to-hidden
             'b_sublstm_x_in': self.sub_lstm.input_w_x.bias, # biases - subLSTM layer, input-to-hidden
             'w_sublstm_h_in': self.sub_lstm.input_w_h.weight, # weights - subLSTM layer, hidden-to-hidden
             'b_sublstm_h_in': self.sub_lstm.input_w_h.bias, # biases - subLSTM layer, hidden-to-hidden
             'w_l1': self.l1.weight, # weights - linear layer
             'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload subLSTM layer
        self.sub_lstm = subLSTM_layer(self.input_size, self.hidden_size)
        self.sub_lstm.f_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_f']).to(torch.float32).to(device))
        self.sub_lstm.f_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_f']).to(torch.float32).to(device))
        self.sub_lstm.f_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_f']).to(torch.float32).to(device))
        self.sub_lstm.f_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_f']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_i']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_o']).to(torch.float32).to(device))
        self.sub_lstm.input_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_in']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x, h_t, c_t, model_params):
        w_sublstm_x_f = model_params['w_sublstm_x_f']
        b_sublstm_x_f = model_params['b_sublstm_x_f']
        w_sublstm_h_f = model_params['w_sublstm_h_f']
        b_sublstm_h_f = model_params['b_sublstm_h_f']
        w_sublstm_x_i = model_params['w_sublstm_x_i']
        b_sublstm_x_i = model_params['b_sublstm_x_i']
        w_sublstm_h_i = model_params['w_sublstm_h_i']
        b_sublstm_h_i = model_params['b_sublstm_h_i']
        w_sublstm_x_o = model_params['w_sublstm_x_o']
        b_sublstm_x_o = model_params['b_sublstm_x_o']
        w_sublstm_h_o = model_params['w_sublstm_h_o']
        b_sublstm_h_o = model_params['b_sublstm_h_o']
        w_sublstm_x_in = model_params['w_sublstm_x_in']
        b_sublstm_x_in = model_params['b_sublstm_x_in']
        w_sublstm_h_in = model_params['w_sublstm_h_in']
        b_sublstm_h_in = model_params['b_sublstm_h_in']
        
        w_l1 = model_params['w_l1']
        b_l1 = model_params['b_l1']
        
        i_t_pre_act = np.dot(w_sublstm_x_i, x) + b_sublstm_x_i[:, None] + np.dot(w_sublstm_h_i, h_t) + b_sublstm_h_i[:, None]
        i_t = 1/(1 + np.exp(-i_t_pre_act))
        
        f_t_pre_act = np.dot(w_sublstm_x_f, x) + b_sublstm_x_f[:, None] + np.dot(w_sublstm_h_f, h_t) + b_sublstm_h_f[:, None]
        f_t = 1/(1 + np.exp(-f_t_pre_act))
        
        out_t_pre_act = np.dot(w_sublstm_x_o, x) + b_sublstm_x_o[:, None] + np.dot(w_sublstm_h_o, h_t) + \
            b_sublstm_h_o[:, None]
        out_t = 1/(1 + np.exp(-out_t_pre_act))

        g_t_pre_act = np.dot(w_sublstm_x_in, x) + b_sublstm_x_in[:, None] + np.dot(w_sublstm_h_in, h_t) + \
            b_sublstm_h_in[:, None]
        g_t = 1/(1 + np.exp(-g_t_pre_act))
        
        c_t = np.multiply(f_t, c_t) + g_t - i_t
        
        h_t = 1/(1 + np.exp(-c_t)) - out_t
        
        o_t_pre_act = np.dot(w_l1, h_t) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
            
        return o_t, o_t_pre_act, h_t, c_t, out_t, out_t_pre_act, g_t, g_t_pre_act, f_t, f_t_pre_act, i_t, i_t_pre_act    
    
    def np_forward_loop(self, inputs):
        np_model_params = {'w_sublstm_x_f': self.sub_lstm.f_gate_w_x.weight.detach().cpu().numpy(), 
                           'b_sublstm_x_f': self.sub_lstm.f_gate_w_x.bias.detach().cpu().numpy(), 
                           'w_sublstm_h_f': self.sub_lstm.f_gate_w_h.weight.detach().cpu().numpy(),
                           'b_sublstm_h_f': self.sub_lstm.f_gate_w_h.bias.detach().cpu().numpy(), 
                           'w_sublstm_x_i': self.sub_lstm.i_gate_w_x.weight.detach().cpu().numpy(), 
                           'b_sublstm_x_i': self.sub_lstm.i_gate_w_x.bias.detach().cpu().numpy(),
                           'w_sublstm_h_i': self.sub_lstm.i_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_i': self.sub_lstm.i_gate_w_h.bias.detach().cpu().numpy(), 
                           'w_sublstm_x_o': self.sub_lstm.o_gate_w_x.weight.detach().cpu().numpy(),
                           'b_sublstm_x_o': self.sub_lstm.o_gate_w_x.bias.detach().cpu().numpy(), 
                           'w_sublstm_h_o': self.sub_lstm.o_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_o': self.sub_lstm.o_gate_w_h.bias.detach().cpu().numpy(),
                           'w_sublstm_x_in': self.sub_lstm.input_w_x.weight.detach().cpu().numpy(), 
                           'b_sublstm_x_in': self.sub_lstm.input_w_x.bias.detach().cpu().numpy(), 
                           'w_sublstm_h_in': self.sub_lstm.input_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_in': self.sub_lstm.input_w_h.bias.detach().cpu().numpy(), 
                           'w_l1': self.l1.weight.detach().cpu().numpy(), 'b_l1': self.l1.bias.detach().cpu().numpy()}
        
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            #print(t)
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, c, out, out_pre_act, g, g_pre_act, f, f_pre_act, i, i_pre_act = self.np_forward(input_t, h, c,
                                                                                                             np_model_params)
            
            if t == 0:
                i_pre_act_trace = np.array(i_pre_act)
                i_trace = np.array(i)
                f_pre_act_trace = np.array(f_pre_act)
                f_trace = np.array(f)
                g_pre_act_trace = np.array(g_pre_act)
                g_trace = np.array(g)
                out_pre_act_trace = np.array(out_pre_act)
                out_trace = np.array(out)
                c_trace = np.array(c)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                i_pre_act_trace = np.append(i_pre_act_trace, i_pre_act, axis=1)
                i_trace = np.append(i_trace, i, axis=1)
                f_pre_act_trace = np.append(f_pre_act_trace, f_pre_act, axis=1)
                f_trace = np.append(f_trace, f, axis=1)
                g_pre_act_trace = np.append(g_pre_act_trace, g_pre_act, axis=1)
                g_trace = np.append(g_trace, g, axis=1)
                out_pre_act_trace = np.append(out_pre_act_trace, out_pre_act, axis=1)
                out_trace = np.append(out_trace, out, axis=1)
                c_trace = np.append(c_trace, c, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'i_pre_act': i_pre_act_trace, 'i': i_trace, 'f_pre_act': f_pre_act_trace, 'f': f_trace,
                  'g_pre_act': g_pre_act_trace, 'g': g_trace, 'out_pre_act': out_pre_act_trace, 'out': out_trace,
                  'c': c_trace, 'h': h_trace, 'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces

#############################################################################################################################

class fix_subLSTM_layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(fix_subLSTM_layer, self).__init__()
        
        self.f_gate = nn.Parameter(torch.Tensor(hidden_size))
        self.i_gate_w_x = nn.Linear(input_size, hidden_size)
        self.i_gate_w_h = nn.Linear(hidden_size, hidden_size)
        self.o_gate_w_x = nn.Linear(input_size, hidden_size)
        self.o_gate_w_h = nn.Linear(hidden_size, hidden_size)
        
        self.input_w_x = nn.Linear(input_size, hidden_size)
        self.input_w_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, h_0, c_0):
        f_gate = self.f_gate
        f_gate = f_gate.clamp(0, 1)
        i_gate = torch.sigmoid(self.i_gate_w_x(x) + self.i_gate_w_h(h_0))
        o_gate = torch.sigmoid(self.o_gate_w_x(x) + self.o_gate_w_h(h_0))
        
        candidate_c = torch.sigmoid(self.input_w_x(x) + self.input_w_h(h_0))
        
        c_t = torch.multiply(f_gate, c_0) + candidate_c - i_gate
        
        h_t = torch.sigmoid(c_t) - o_gate
        out = h_t
        
        return out, h_t, c_t

class pop_fix_subLSTM(nn.Module):
    def __init__(self, params):
        super(pop_fix_subLSTM, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        # subLSTM layer (input-to-hidden)
        self.sub_lstm = fix_subLSTM_layer(self.input_size, self.hidden_size)
        self.l1 = nn.Linear(self.hidden_size, self.output_size) # linear layer (hidden-to-output)

        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, c_0, stage):
        # forward pass of data through network
        # input x dimensions -> ()
        
        if self.dropout is True and stage == 'train':
            out, h_n, c_n = self.sub_lstm(self.input_dropout(x), h_0, c_0)
            out = torch.sigmoid(self.l1(self.hidden_dropout(out)))
        else:
            out, h_n, c_n = self.sub_lstm(x, h_0, c_0)
            out = torch.sigmoid(self.l1(out))
            
        return out, h_n, c_n
    
    def forward_loop(self, inputs, target_data, stage):
        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device)
        c_n = torch.zeros((1, self.hidden_size), device=device)
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
                c_n = torch.zeros((1, self.hidden_size), device=device)
        
            input_t = inputs[:, :, t].view(-1, self.input_size)
            
            # forward
            output, h_n, c_n = self.forward(input_t, h_n, c_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
        
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.sub_lstm.i_gate_w_x.weight.norm(p=1) + self.sub_lstm.i_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.o_gate_w_x.weight.norm(p=1) + self.sub_lstm.o_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.input_w_x.weight.norm(p=1) + self.sub_lstm.input_w_h.weight.norm(p=1) + \
                           self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):        
        model_info = {
             'sublstm_f': self.sub_lstm.f_gate, # value of fixed f gate
             'w_sublstm_x_i': self.sub_lstm.i_gate_w_x.weight, # weights - subLSTM layer, input-to-i gate
             'b_sublstm_x_i': self.sub_lstm.i_gate_w_x.bias, # biases - subLSTM layer, input-to-i gate
             'w_sublstm_h_i': self.sub_lstm.i_gate_w_h.weight, # weights - subLSTM layer, hidden-to-i gate
             'b_sublstm_h_i': self.sub_lstm.i_gate_w_h.bias, # biases - subLSTM layer, hidden-to-i gate
             'w_sublstm_x_o': self.sub_lstm.o_gate_w_x.weight, # weights - subLSTM layer, input-to-o gate
             'b_sublstm_x_o': self.sub_lstm.o_gate_w_x.bias, # biases - subLSTM layer, input-to-o gate
             'w_sublstm_h_o': self.sub_lstm.o_gate_w_h.weight, # weights - subLSTM layer, hidden-to-o gate
             'b_sublstm_h_o': self.sub_lstm.o_gate_w_h.bias, # biases - subLSTM layer, hidden-to-o gate
             'w_sublstm_x_in': self.sub_lstm.input_w_x.weight, # weights - subLSTM layer, input-to-hidden
             'b_sublstm_x_in': self.sub_lstm.input_w_x.bias, # biases - subLSTM layer, input-to-hidden
             'w_sublstm_h_in': self.sub_lstm.input_w_h.weight, # weights - subLSTM layer, hidden-to-hidden
             'b_sublstm_h_in': self.sub_lstm.input_w_h.bias, # biases - subLSTM layer, hidden-to-hidden
             'w_l1': self.l1.weight, # weights - linear layer
             'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload subLSTM layer
        self.sub_lstm = fix_subLSTM_layer(self.input_size, self.hidden_size)
        self.sub_lstm.f_gate = nn.Parameter(torch.tensor(info['sublstm_f']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_i']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_o']).to(torch.float32).to(device))
        self.sub_lstm.input_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_in']).to(torch.float32).to(device))

        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x, h_t, c_t, model_params):
        sublstm_f = model_params['sublstm_f']
        w_sublstm_x_i = model_params['w_sublstm_x_i']
        b_sublstm_x_i = model_params['b_sublstm_x_i']
        w_sublstm_h_i = model_params['w_sublstm_h_i']
        b_sublstm_h_i = model_params['b_sublstm_h_i']
        w_sublstm_x_o = model_params['w_sublstm_x_o']
        b_sublstm_x_o = model_params['b_sublstm_x_o']
        w_sublstm_h_o = model_params['w_sublstm_h_o']
        b_sublstm_h_o = model_params['b_sublstm_h_o']
        w_sublstm_x_in = model_params['w_sublstm_x_in']
        b_sublstm_x_in = model_params['b_sublstm_x_in']
        w_sublstm_h_in = model_params['w_sublstm_h_in']
        b_sublstm_h_in = model_params['b_sublstm_h_in']
        
        w_l1 = model_params['w_l1']
        b_l1 = model_params['b_l1']
        
        i_t_pre_act = np.dot(w_sublstm_x_i, x) + b_sublstm_x_i[:, None] + np.dot(w_sublstm_h_i, h_t) + b_sublstm_h_i[:, None]
        i_t = 1/(1 + np.exp(-i_t_pre_act))

        f_t = sublstm_f[:, None]
        f_t = np.clip(f_t, 0, 1)
        
        out_t_pre_act = np.dot(w_sublstm_x_o, x) + b_sublstm_x_o[:, None] + np.dot(w_sublstm_h_o, h_t) + \
            b_sublstm_h_o[:, None]
        out_t = 1/(1 + np.exp(-out_t_pre_act))

        g_t_pre_act = np.dot(w_sublstm_x_in, x) + b_sublstm_x_in[:, None] + np.dot(w_sublstm_h_in, h_t) + \
            b_sublstm_h_in[:, None]
        g_t = 1/(1 + np.exp(-g_t_pre_act))

        c_t = np.multiply(f_t, c_t) + g_t - i_t
        
        h_t = 1/(1 + np.exp(-c_t)) - out_t
        
        o_t_pre_act = np.dot(w_l1, h_t) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
            
        return o_t, o_t_pre_act, h_t, c_t, out_t, out_t_pre_act, g_t, g_t_pre_act, f_t, i_t, i_t_pre_act    
    
    def np_forward_loop(self, inputs):
        np_model_params = {'sublstm_f': self.sub_lstm.f_gate.detach().cpu().numpy(),
                           'w_sublstm_x_i': self.sub_lstm.i_gate_w_x.weight.detach().cpu().numpy(), 
                           'b_sublstm_x_i': self.sub_lstm.i_gate_w_x.bias.detach().cpu().numpy(),
                           'w_sublstm_h_i': self.sub_lstm.i_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_i': self.sub_lstm.i_gate_w_h.bias.detach().cpu().numpy(), 
                           'w_sublstm_x_o': self.sub_lstm.o_gate_w_x.weight.detach().cpu().numpy(),
                           'b_sublstm_x_o': self.sub_lstm.o_gate_w_x.bias.detach().cpu().numpy(), 
                           'w_sublstm_h_o': self.sub_lstm.o_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_o': self.sub_lstm.o_gate_w_h.bias.detach().cpu().numpy(),
                           'w_sublstm_x_in': self.sub_lstm.input_w_x.weight.detach().cpu().numpy(), 
                           'b_sublstm_x_in': self.sub_lstm.input_w_x.bias.detach().cpu().numpy(), 
                           'w_sublstm_h_in': self.sub_lstm.input_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_in': self.sub_lstm.input_w_h.bias.detach().cpu().numpy(), 
                           'w_l1': self.l1.weight.detach().cpu().numpy(), 'b_l1': self.l1.bias.detach().cpu().numpy()}
        
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            #print(t)
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, c, out, out_pre_act, g, g_pre_act, f, i, i_pre_act = self.np_forward(input_t, h, c, 
                                                                                                  np_model_params)

            if t == 0:
                i_pre_act_trace = np.array(i_pre_act)
                i_trace = np.array(i)
                f_trace = np.array(f)
                g_pre_act_trace = np.array(g_pre_act)
                g_trace = np.array(g)
                out_pre_act_trace = np.array(out_pre_act)
                out_trace = np.array(out)
                c_trace = np.array(c)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                i_pre_act_trace = np.append(i_pre_act_trace, i_pre_act, axis=1)
                i_trace = np.append(i_trace, i, axis=1)
                f_trace = np.append(f_trace, f, axis=1)
                g_pre_act_trace = np.append(g_pre_act_trace, g_pre_act, axis=1)
                g_trace = np.append(g_trace, g, axis=1)
                out_pre_act_trace = np.append(out_pre_act_trace, out_pre_act, axis=1)
                out_trace = np.append(out_trace, out, axis=1)
                c_trace = np.append(c_trace, c, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'i_pre_act': i_pre_act_trace, 'i': i_trace, 'f': f_trace, 'g_pre_act': g_pre_act_trace, 'g': g_trace, 
                  'out_pre_act': out_pre_act_trace, 'out': out_trace, 'c': c_trace, 'h': h_trace, 
                  'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces
    
#############################################################################################################################

class pop_f_LSTM(nn.Module):
    def __init__(self, params):
        super(pop_f_LSTM, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        self.t_gate_w_x = nn.Linear(self.input_size, self.input_size)
        self.t_gate_w_h = nn.Linear(self.hidden_size, self.input_size)
        self.sub_lstm = subLSTM_layer(self.input_size, self.hidden_size)
        self.l1 = nn.Linear(self.hidden_size, self.output_size) # linear layer (hidden-to-output)

        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        if 't_gate_memory' in params:    
            self.t_gate_memory = params['t_gate_memory']
        else:
            self.t_gate_memory = 'hidden'
        
        if 'split' in params:
            self.split = params['split']
        else:
            self.split = 1
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, c_0, stage):
        # forward pass of data through network
        
        if self.dropout is True and stage == 'train':
            out, h_n, c_n = self.sub_lstm(self.input_dropout(x), h_0, c_0)
            out = torch.sigmoid(self.l1(self.hidden_dropout(out)))
        else:
            if self.t_gate_memory == 'hidden':
                t_gate = torch.sigmoid(self.t_gate_w_x(x) + self.t_gate_w_h(h_0))
            elif self.t_gate_memory == 'cell':
                t_gate = torch.sigmoid(self.t_gate_w_x(x) + self.t_gate_w_h(c_0))
                
            x_t_gate = torch.multiply(t_gate, x)
            out, h_n, c_n = self.sub_lstm(x_t_gate, h_0, c_0)
            #out = torch.sigmoid(self.l1(out))
            
            out = self.l1(out)

        return out, h_n, c_n
    
    def forward_loop(self, inputs, target_data, stage):
        compute_grad = False
        
        if stage == 'train':
            stim_size = self.stim_size
        elif stage == 'val' or stage == 'test':
            n_data = int(np.floor(inputs.size(2)/(self.stim_size/self.split)))
            stim_size = int(inputs.size(2)/n_data)
        
        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        if compute_grad:
            hidden = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device)
        c_n = torch.zeros((1, self.hidden_size), device=device)
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
                c_n = torch.zeros((1, self.hidden_size), device=device)
            
            input_t = inputs[:, :, t].view(-1, self.input_size)
            
            # forward
            output, h_n, c_n = self.forward(input_t, h_n, c_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
            
            if compute_grad:
                hidden = torch.cat((hidden, h_n), dim=0)

        if compute_grad:
            return outputs, targets, hidden
        else:
            return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.t_gate_w_x.weight.norm(p=1) + self.t_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.f_gate_w_x.weight.norm(p=1) + self.sub_lstm.f_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.i_gate_w_x.weight.norm(p=1) + self.sub_lstm.i_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.o_gate_w_x.weight.norm(p=1) + self.sub_lstm.o_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.input_w_x.weight.norm(p=1) + self.sub_lstm.input_w_h.weight.norm(p=1) + \
                           self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):        
        model_info = {
            'w_x_t': self.t_gate_w_x.weight, # weights - input-to-t gate
            'b_x_t': self.t_gate_w_x.bias, # biases - input-to-t gate
            'w_h_t': self.t_gate_w_h.weight, # weights - hidden-to-t gate
            'b_h_t': self.t_gate_w_h.bias, # biases - hidden-to-t gate
            'w_sublstm_x_f': self.sub_lstm.f_gate_w_x.weight, # weights - subLSTM layer, input-to-f gate
            'b_sublstm_x_f': self.sub_lstm.f_gate_w_x.bias, # biases - subLSTM layer, input-to-f gate
            'w_sublstm_h_f': self.sub_lstm.f_gate_w_h.weight, # weights - subLSTM layer, hidden-to-f gate
            'b_sublstm_h_f': self.sub_lstm.f_gate_w_h.bias, # biases - subLSTM layer, hidden-to-f gate
            'w_sublstm_x_i': self.sub_lstm.i_gate_w_x.weight, # weights - subLSTM layer, input-to-i gate
            'b_sublstm_x_i': self.sub_lstm.i_gate_w_x.bias, # biases - subLSTM layer, input-to-i gate
            'w_sublstm_h_i': self.sub_lstm.i_gate_w_h.weight, # weights - subLSTM layer, hidden-to-i gate
            'b_sublstm_h_i': self.sub_lstm.i_gate_w_h.bias, # biases - subLSTM layer, hidden-to-i gate
            'w_sublstm_x_o': self.sub_lstm.o_gate_w_x.weight, # weights - subLSTM layer, input-to-o gate
            'b_sublstm_x_o': self.sub_lstm.o_gate_w_x.bias, # biases - subLSTM layer, input-to-o gate
            'w_sublstm_h_o': self.sub_lstm.o_gate_w_h.weight, # weights - subLSTM layer, hidden-to-o gate
            'b_sublstm_h_o': self.sub_lstm.o_gate_w_h.bias, # biases - subLSTM layer, hidden-to-o gate
            'w_sublstm_x_in': self.sub_lstm.input_w_x.weight, # weights - subLSTM layer, input-to-hidden
            'b_sublstm_x_in': self.sub_lstm.input_w_x.bias, # biases - subLSTM layer, input-to-hidden
            'w_sublstm_h_in': self.sub_lstm.input_w_h.weight, # weights - subLSTM layer, hidden-to-hidden
            'b_sublstm_h_in': self.sub_lstm.input_w_h.bias, # biases - subLSTM layer, hidden-to-hidden
            'w_l1': self.l1.weight, # weights - linear layer
            'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload t gate layer
        self.t_gate_w_x = nn.Linear(self.input_size, self.input_size)
        self.t_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_x_t']).to(torch.float32).to(device))
        self.t_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_x_t']).to(torch.float32).to(device))
        
        self.t_gate_w_h = nn.Linear(self.hidden_size, self.input_size)
        self.t_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_h_t']).to(torch.float32).to(device))
        self.t_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_h_t']).to(torch.float32).to(device))
        
        # reload subLSTM layer
        self.sub_lstm = subLSTM_layer(self.input_size, self.hidden_size)
        self.sub_lstm.f_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_f']).to(torch.float32).to(device))
        self.sub_lstm.f_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_f']).to(torch.float32).to(device))
        self.sub_lstm.f_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_f']).to(torch.float32).to(device))
        self.sub_lstm.f_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_f']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_i']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_o']).to(torch.float32).to(device))
        self.sub_lstm.input_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_in']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x, h_t, c_t, model_params):
        w_x_t = model_params['w_x_t']
        b_x_t = model_params['b_x_t']
        w_h_t = model_params['w_h_t']
        b_h_t = model_params['b_h_t']
        
        w_sublstm_x_f = model_params['w_sublstm_x_f']
        b_sublstm_x_f = model_params['b_sublstm_x_f']
        w_sublstm_h_f = model_params['w_sublstm_h_f']
        b_sublstm_h_f = model_params['b_sublstm_h_f']
        w_sublstm_x_i = model_params['w_sublstm_x_i']
        b_sublstm_x_i = model_params['b_sublstm_x_i']
        w_sublstm_h_i = model_params['w_sublstm_h_i']
        b_sublstm_h_i = model_params['b_sublstm_h_i']
        w_sublstm_x_o = model_params['w_sublstm_x_o']
        b_sublstm_x_o = model_params['b_sublstm_x_o']
        w_sublstm_h_o = model_params['w_sublstm_h_o']
        b_sublstm_h_o = model_params['b_sublstm_h_o']
        w_sublstm_x_in = model_params['w_sublstm_x_in']
        b_sublstm_x_in = model_params['b_sublstm_x_in']
        w_sublstm_h_in = model_params['w_sublstm_h_in']
        b_sublstm_h_in = model_params['b_sublstm_h_in']
        
        w_l1 = model_params['w_l1']
        b_l1 = model_params['b_l1']
        
        if self.t_gate_memory == 'hidden':
            t_t_pre_act = np.dot(w_x_t, x) + b_x_t[:, None] + np.dot(w_h_t, h_t) + b_h_t[:, None]
        elif self.t_gate_memory == 'cell':
            t_t_pre_act = np.dot(w_x_t, x) + b_x_t[:, None] + np.dot(w_h_t, c_t) + b_h_t[:, None]
        t_t = 1/(1 + np.exp(-t_t_pre_act))
        
        x_t_gate = np.multiply(t_t, x)
        
        i_t_pre_act = np.dot(w_sublstm_x_i, x_t_gate) + b_sublstm_x_i[:, None] + np.dot(w_sublstm_h_i, h_t) + \
            b_sublstm_h_i[:, None]
        i_t = 1/(1 + np.exp(-i_t_pre_act))
        
        f_t_pre_act = np.dot(w_sublstm_x_f, x_t_gate) + b_sublstm_x_f[:, None] + np.dot(w_sublstm_h_f, h_t) + \
            b_sublstm_h_f[:, None]
        f_t = 1/(1 + np.exp(-f_t_pre_act))
        
        out_t_pre_act = np.dot(w_sublstm_x_o, x_t_gate) + b_sublstm_x_o[:, None] + np.dot(w_sublstm_h_o, h_t) + \
            b_sublstm_h_o[:, None]
        out_t = 1/(1 + np.exp(-out_t_pre_act))

        g_t_pre_act = np.dot(w_sublstm_x_in, x_t_gate) + b_sublstm_x_in[:, None] + np.dot(w_sublstm_h_in, h_t) + \
            b_sublstm_h_in[:, None]
        g_t = 1/(1 + np.exp(-g_t_pre_act))
        
        c_t = np.multiply(f_t, c_t) + g_t - i_t
        
        h_t = 1/(1 + np.exp(-c_t)) - out_t
        
        o_t_pre_act = np.dot(w_l1, h_t) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
            
        return o_t, o_t_pre_act, h_t, c_t, out_t, out_t_pre_act, g_t, g_t_pre_act, f_t, f_t_pre_act, i_t, i_t_pre_act, \
            x_t_gate, t_t, t_t_pre_act
    
    def np_forward_loop(self, inputs):
        np_model_params = {'w_x_t': self.t_gate_w_x.weight.detach().cpu().numpy(),
                           'b_x_t': self.t_gate_w_x.bias.detach().cpu().numpy(),
                           'w_h_t': self.t_gate_w_h.weight.detach().cpu().numpy(),
                           'b_h_t': self.t_gate_w_h.bias.detach().cpu().numpy(),
                           'w_sublstm_x_f': self.sub_lstm.f_gate_w_x.weight.detach().cpu().numpy(), 
                           'b_sublstm_x_f': self.sub_lstm.f_gate_w_x.bias.detach().cpu().numpy(), 
                           'w_sublstm_h_f': self.sub_lstm.f_gate_w_h.weight.detach().cpu().numpy(),
                           'b_sublstm_h_f': self.sub_lstm.f_gate_w_h.bias.detach().cpu().numpy(), 
                           'w_sublstm_x_i': self.sub_lstm.i_gate_w_x.weight.detach().cpu().numpy(), 
                           'b_sublstm_x_i': self.sub_lstm.i_gate_w_x.bias.detach().cpu().numpy(),
                           'w_sublstm_h_i': self.sub_lstm.i_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_i': self.sub_lstm.i_gate_w_h.bias.detach().cpu().numpy(), 
                           'w_sublstm_x_o': self.sub_lstm.o_gate_w_x.weight.detach().cpu().numpy(),
                           'b_sublstm_x_o': self.sub_lstm.o_gate_w_x.bias.detach().cpu().numpy(), 
                           'w_sublstm_h_o': self.sub_lstm.o_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_o': self.sub_lstm.o_gate_w_h.bias.detach().cpu().numpy(),
                           'w_sublstm_x_in': self.sub_lstm.input_w_x.weight.detach().cpu().numpy(), 
                           'b_sublstm_x_in': self.sub_lstm.input_w_x.bias.detach().cpu().numpy(), 
                           'w_sublstm_h_in': self.sub_lstm.input_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_in': self.sub_lstm.input_w_h.bias.detach().cpu().numpy(), 
                           'w_l1': self.l1.weight.detach().cpu().numpy(), 'b_l1': self.l1.bias.detach().cpu().numpy()}
        
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            #print(t)
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, c, out, out_pre_act, g, g_pre_act, f, f_pre_act, i, i_pre_act, x_t_gated, t_gate, \
                t_gate_pre_act = self.np_forward(input_t, h, c, np_model_params)

            if t == 0:
                t_pre_act_trace = np.array(t_gate_pre_act)
                t_trace = np.array(t_gate)
                x_t_gated_trace = np.array(x_t_gated)
                i_pre_act_trace = np.array(i_pre_act)
                i_trace = np.array(i)
                f_pre_act_trace = np.array(f_pre_act)
                f_trace = np.array(f)
                g_pre_act_trace = np.array(g_pre_act)
                g_trace = np.array(g)
                out_pre_act_trace = np.array(out_pre_act)
                out_trace = np.array(out)
                c_trace = np.array(c)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                t_pre_act_trace = np.append(t_pre_act_trace, t_gate_pre_act, axis=1)
                t_trace = np.append(t_trace, t_gate, axis=1)
                x_t_gated_trace = np.append(x_t_gated_trace, x_t_gated, axis=1)
                i_pre_act_trace = np.append(i_pre_act_trace, i_pre_act, axis=1)
                i_trace = np.append(i_trace, i, axis=1)
                f_pre_act_trace = np.append(f_pre_act_trace, f_pre_act, axis=1)
                f_trace = np.append(f_trace, f, axis=1)
                g_pre_act_trace = np.append(g_pre_act_trace, g_pre_act, axis=1)
                g_trace = np.append(g_trace, g, axis=1)
                out_pre_act_trace = np.append(out_pre_act_trace, out_pre_act, axis=1)
                out_trace = np.append(out_trace, out, axis=1)
                c_trace = np.append(c_trace, c, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'t_pre_act': t_pre_act_trace, 't': t_trace, 'x_t_gated': x_t_gated_trace, 'i_pre_act': i_pre_act_trace, 
                  'i': i_trace, 'f_pre_act': f_pre_act_trace, 'f': f_trace, 'g_pre_act': g_pre_act_trace, 'g': g_trace, 
                  'out_pre_act': out_pre_act_trace, 'out': out_trace, 'c': c_trace, 'h': h_trace, 
                  'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces
    
#############################################################################################################################

class pop_rc_LSTM(nn.Module):
    def __init__(self, params):
        super(pop_rc_LSTM, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.added_layers_size = 256
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        self.t1_gate_w_x = nn.Linear(self.input_size, self.input_size)
        self.t1_gate_w_h = nn.Linear(self.hidden_size, self.input_size)
        self.added_l1 = nn.Linear(self.input_size, self.added_layers_size)
        self.sub_lstm = subLSTM_layer(self.added_layers_size, self.hidden_size)
        self.l1 = nn.Linear(self.hidden_size, self.output_size) # linear layer (hidden-to-output)

        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        if 'stim_size' in params:
            self.stim_size = params['stim_size'] # stimulus length (in time bins)
        if 'resp_size' in params:
            self.stim_size = params['resp_size'] + 1
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, c_0, stage):
        # forward pass of data through network
        
        t1_gate = torch.sigmoid(self.t1_gate_w_x(x) + self.t1_gate_w_h(h_0))
        x_t_gated = torch.multiply(t1_gate, x)
        x = torch.sigmoid(self.added_l1(x_t_gated))
        out, h_n, c_n = self.sub_lstm(x, h_0, c_0)
        #out = torch.sigmoid(self.l1(out))
        
        out = self.l1(out)
        
        return out, h_n, c_n
    
    def forward_loop(self, inputs, target_data, stage):
        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device)
        c_n = torch.zeros((1, self.hidden_size), device=device)
        
        for t in range(inputs.size(-1)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
                c_n = torch.zeros((1, self.hidden_size), device=device)

            if len(inputs.shape) == 3:
                input_t = inputs[:, :, t].view(-1, self.input_size)
            elif len(inputs.shape) == 2:
                inputs
                input_t = inputs[:, t].view(-1, self.input_size)
            
            # forward
            output, h_n, c_n = self.forward(input_t, h_n, c_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
        
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.t1_gate_w_x.weight.norm(p=1) + self.t1_gate_w_h.weight.norm(p=1) + \
                           self.added_l1.weight.norm(p=1) + \
                           self.sub_lstm.f_gate_w_x.weight.norm(p=1) + self.sub_lstm.f_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.i_gate_w_x.weight.norm(p=1) + self.sub_lstm.i_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.o_gate_w_x.weight.norm(p=1) + self.sub_lstm.o_gate_w_h.weight.norm(p=1) + \
                           self.sub_lstm.input_w_x.weight.norm(p=1) + self.sub_lstm.input_w_h.weight.norm(p=1) + \
                           self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):        
        model_info = {
            'w_x_t1': self.t1_gate_w_x.weight, # weights - input-to-t1 gate
            'b_x_t1': self.t1_gate_w_x.bias, # biases - input-to-t1 gate
            'w_h_t1': self.t1_gate_w_h.weight, # weights - hidden-to-t1 gate
            'b_h_t1': self.t1_gate_w_h.bias, # biases - hidden-to-t1 gate
            'w_added1': self.added_l1.weight, # weights - added linear layer
            'b_added1': self.added_l1.bias, # biases - added linear layer
            'w_sublstm_x_f': self.sub_lstm.f_gate_w_x.weight, # weights - subLSTM layer, input-to-f gate
            'b_sublstm_x_f': self.sub_lstm.f_gate_w_x.bias, # biases - subLSTM layer, input-to-f gate
            'w_sublstm_h_f': self.sub_lstm.f_gate_w_h.weight, # weights - subLSTM layer, hidden-to-f gate
            'b_sublstm_h_f': self.sub_lstm.f_gate_w_h.bias, # biases - subLSTM layer, hidden-to-f gate
            'w_sublstm_x_i': self.sub_lstm.i_gate_w_x.weight, # weights - subLSTM layer, input-to-i gate
            'b_sublstm_x_i': self.sub_lstm.i_gate_w_x.bias, # biases - subLSTM layer, input-to-i gate
            'w_sublstm_h_i': self.sub_lstm.i_gate_w_h.weight, # weights - subLSTM layer, hidden-to-i gate
            'b_sublstm_h_i': self.sub_lstm.i_gate_w_h.bias, # biases - subLSTM layer, hidden-to-i gate
            'w_sublstm_x_o': self.sub_lstm.o_gate_w_x.weight, # weights - subLSTM layer, input-to-o gate
            'b_sublstm_x_o': self.sub_lstm.o_gate_w_x.bias, # biases - subLSTM layer, input-to-o gate
            'w_sublstm_h_o': self.sub_lstm.o_gate_w_h.weight, # weights - subLSTM layer, hidden-to-o gate
            'b_sublstm_h_o': self.sub_lstm.o_gate_w_h.bias, # biases - subLSTM layer, hidden-to-o gate
            'w_sublstm_x_in': self.sub_lstm.input_w_x.weight, # weights - subLSTM layer, input-to-hidden
            'b_sublstm_x_in': self.sub_lstm.input_w_x.bias, # biases - subLSTM layer, input-to-hidden
            'w_sublstm_h_in': self.sub_lstm.input_w_h.weight, # weights - subLSTM layer, hidden-to-hidden
            'b_sublstm_h_in': self.sub_lstm.input_w_h.bias, # biases - subLSTM layer, hidden-to-hidden
            'w_l1': self.l1.weight, # weights - linear layer
            'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload t1 gate layer
        self.t1_gate_w_x = nn.Linear(self.input_size, self.input_size)
        self.t1_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_x_t1']).to(torch.float32).to(device))
        self.t1_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_x_t1']).to(torch.float32).to(device))
        
        self.t1_gate_w_h = nn.Linear(self.hidden_size, self.input_size)
        self.t1_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_h_t1']).to(torch.float32).to(device))
        self.t1_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_h_t1']).to(torch.float32).to(device))
        
        # reload added linear layer
        self.added_l1 = nn.Linear(self.input_size, self.added_layers_size)
        self.added_l1.weight = nn.Parameter(torch.tensor(info['w_added1']).to(torch.float32).to(device))
        self.added_l1.bias = nn.Parameter(torch.tensor(info['b_added1']).to(torch.float32).to(device))
        
        # reload subLSTM layer
        self.sub_lstm = subLSTM_layer(self.input_size, self.hidden_size)
        self.sub_lstm.f_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_f']).to(torch.float32).to(device))
        self.sub_lstm.f_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_f']).to(torch.float32).to(device))
        self.sub_lstm.f_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_f']).to(torch.float32).to(device))
        self.sub_lstm.f_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_f']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_i']).to(torch.float32).to(device))
        self.sub_lstm.i_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_i']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_o']).to(torch.float32).to(device))
        self.sub_lstm.o_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_o']).to(torch.float32).to(device))
        self.sub_lstm.input_w_x.weight = nn.Parameter(torch.tensor(info['w_sublstm_x_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_x.bias = nn.Parameter(torch.tensor(info['b_sublstm_x_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_h.weight = nn.Parameter(torch.tensor(info['w_sublstm_h_in']).to(torch.float32).to(device))
        self.sub_lstm.input_w_h.bias = nn.Parameter(torch.tensor(info['b_sublstm_h_in']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x, h_t, c_t, model_params):
        w_x_t1 = model_params['w_x_t1']
        b_x_t1 = model_params['b_x_t1']
        w_h_t1 = model_params['w_h_t1']
        b_h_t1 = model_params['b_h_t1']
        
        w_added1 = model_params['w_added1']
        b_added1 = model_params['b_added1']
        
        w_sublstm_x_f = model_params['w_sublstm_x_f']
        b_sublstm_x_f = model_params['b_sublstm_x_f']
        w_sublstm_h_f = model_params['w_sublstm_h_f']
        b_sublstm_h_f = model_params['b_sublstm_h_f']
        w_sublstm_x_i = model_params['w_sublstm_x_i']
        b_sublstm_x_i = model_params['b_sublstm_x_i']
        w_sublstm_h_i = model_params['w_sublstm_h_i']
        b_sublstm_h_i = model_params['b_sublstm_h_i']
        w_sublstm_x_o = model_params['w_sublstm_x_o']
        b_sublstm_x_o = model_params['b_sublstm_x_o']
        w_sublstm_h_o = model_params['w_sublstm_h_o']
        b_sublstm_h_o = model_params['b_sublstm_h_o']
        w_sublstm_x_in = model_params['w_sublstm_x_in']
        b_sublstm_x_in = model_params['b_sublstm_x_in']
        w_sublstm_h_in = model_params['w_sublstm_h_in']
        b_sublstm_h_in = model_params['b_sublstm_h_in']
        
        w_l1 = model_params['w_l1']
        b_l1 = model_params['b_l1']
        
        # t gate at input
        t1_t_pre_act = np.dot(w_x_t1, x) + b_x_t1[:, None] + np.dot(w_h_t1, h_t) + b_h_t1[:, None]
        t1_t = 1/(1 + np.exp(-t1_t_pre_act))
        x_t1_gate = np.multiply(t1_t, x)
        
        # added feedforward layer
        x_t1_pre_act = np.dot(w_added1, x_t1_gate) + b_added1[:, None]
        x_t1 = 1/(1 + np.exp(-x_t1_pre_act))
        
        i_t_pre_act = np.dot(w_sublstm_x_i, x_t1) + b_sublstm_x_i[:, None] + np.dot(w_sublstm_h_i, h_t) + \
            b_sublstm_h_i[:, None]
        i_t = 1/(1 + np.exp(-i_t_pre_act))
        
        f_t_pre_act = np.dot(w_sublstm_x_f, x_t1) + b_sublstm_x_f[:, None] + np.dot(w_sublstm_h_f, h_t) + \
            b_sublstm_h_f[:, None]
        f_t = 1/(1 + np.exp(-f_t_pre_act))
        
        out_t_pre_act = np.dot(w_sublstm_x_o, x_t1) + b_sublstm_x_o[:, None] + np.dot(w_sublstm_h_o, h_t) + \
            b_sublstm_h_o[:, None]
        out_t = 1/(1 + np.exp(-out_t_pre_act))

        g_t_pre_act = np.dot(w_sublstm_x_in, x_t1) + b_sublstm_x_in[:, None] + np.dot(w_sublstm_h_in, h_t) + \
            b_sublstm_h_in[:, None]
        g_t = 1/(1 + np.exp(-g_t_pre_act))
        
        c_t = np.multiply(f_t, c_t) + g_t - i_t
        
        h_t = 1/(1 + np.exp(-c_t)) - out_t
        
        o_t_pre_act = np.dot(w_l1, h_t) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
            
        return o_t, o_t_pre_act, h_t, c_t, out_t, out_t_pre_act, g_t, g_t_pre_act, f_t, f_t_pre_act, i_t, i_t_pre_act, \
            x_t1, x_t1_pre_act, x_t1_gate, t1_t, t1_t_pre_act
    
    def np_forward_loop(self, inputs):
        np_model_params = {'w_x_t1': self.t1_gate_w_x.weight.detach().cpu().numpy(),
                           'b_x_t1': self.t1_gate_w_x.bias.detach().cpu().numpy(),
                           'w_h_t1': self.t1_gate_w_h.weight.detach().cpu().numpy(),
                           'b_h_t1': self.t1_gate_w_h.bias.detach().cpu().numpy(),
                           'w_added1': self.added_l1.weight.detach().cpu().numpy(),
                           'b_added1': self.added_l1.bias.detach().cpu().numpy(),
                           'w_sublstm_x_f': self.sub_lstm.f_gate_w_x.weight.detach().cpu().numpy(), 
                           'b_sublstm_x_f': self.sub_lstm.f_gate_w_x.bias.detach().cpu().numpy(), 
                           'w_sublstm_h_f': self.sub_lstm.f_gate_w_h.weight.detach().cpu().numpy(),
                           'b_sublstm_h_f': self.sub_lstm.f_gate_w_h.bias.detach().cpu().numpy(), 
                           'w_sublstm_x_i': self.sub_lstm.i_gate_w_x.weight.detach().cpu().numpy(), 
                           'b_sublstm_x_i': self.sub_lstm.i_gate_w_x.bias.detach().cpu().numpy(),
                           'w_sublstm_h_i': self.sub_lstm.i_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_i': self.sub_lstm.i_gate_w_h.bias.detach().cpu().numpy(), 
                           'w_sublstm_x_o': self.sub_lstm.o_gate_w_x.weight.detach().cpu().numpy(),
                           'b_sublstm_x_o': self.sub_lstm.o_gate_w_x.bias.detach().cpu().numpy(), 
                           'w_sublstm_h_o': self.sub_lstm.o_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_o': self.sub_lstm.o_gate_w_h.bias.detach().cpu().numpy(),
                           'w_sublstm_x_in': self.sub_lstm.input_w_x.weight.detach().cpu().numpy(), 
                           'b_sublstm_x_in': self.sub_lstm.input_w_x.bias.detach().cpu().numpy(), 
                           'w_sublstm_h_in': self.sub_lstm.input_w_h.weight.detach().cpu().numpy(), 
                           'b_sublstm_h_in': self.sub_lstm.input_w_h.bias.detach().cpu().numpy(), 
                           'w_l1': self.l1.weight.detach().cpu().numpy(), 'b_l1': self.l1.bias.detach().cpu().numpy()}
        
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            #print(t)
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, c, out, out_pre_act, g, g_pre_act, f, f_pre_act, i, i_pre_act, x_t1, x_t1_pre_act, \
                x_t1_gated, t1_gate, t1_gate_pre_act = \
                    self.np_forward(input_t, h, c, np_model_params)

            if t == 0:
                t1_pre_act_trace = np.array(t1_gate_pre_act)
                t1_trace = np.array(t1_gate)
                x_t1_gated_trace = np.array(x_t1_gated)
                x_t1_pre_act_trace = np.array(x_t1_pre_act)
                x_t1_trace = np.array(x_t1)
                i_pre_act_trace = np.array(i_pre_act)
                i_trace = np.array(i)
                f_pre_act_trace = np.array(f_pre_act)
                f_trace = np.array(f)
                g_pre_act_trace = np.array(g_pre_act)
                g_trace = np.array(g)
                out_pre_act_trace = np.array(out_pre_act)
                out_trace = np.array(out)
                c_trace = np.array(c)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                t1_pre_act_trace = np.append(t1_pre_act_trace, t1_gate_pre_act, axis=1)
                t1_trace = np.append(t1_trace, t1_gate, axis=1)
                x_t1_gated_trace = np.append(x_t1_gated_trace, x_t1_gated, axis=1)
                x_t1_pre_act_trace = np.append(x_t1_pre_act_trace, x_t1_pre_act, axis=1)
                x_t1_trace = np.append(x_t1_trace, x_t1, axis=1)
                i_pre_act_trace = np.append(i_pre_act_trace, i_pre_act, axis=1)
                i_trace = np.append(i_trace, i, axis=1)
                f_pre_act_trace = np.append(f_pre_act_trace, f_pre_act, axis=1)
                f_trace = np.append(f_trace, f, axis=1)
                g_pre_act_trace = np.append(g_pre_act_trace, g_pre_act, axis=1)
                g_trace = np.append(g_trace, g, axis=1)
                out_pre_act_trace = np.append(out_pre_act_trace, out_pre_act, axis=1)
                out_trace = np.append(out_trace, out, axis=1)
                c_trace = np.append(c_trace, c, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'t1_pre_act': t1_pre_act_trace, 't1': t1_trace, 'x_t1_gated': x_t1_gated_trace, 
                  'x_t1_pre_act': x_t1_pre_act_trace, 'x_t1': x_t1_trace, 'i_pre_act': i_pre_act_trace, 'i': i_trace, 
                  'f_pre_act': f_pre_act_trace, 'f': f_trace, 'g_pre_act': g_pre_act_trace, 'g': g_trace, 
                  'out_pre_act': out_pre_act_trace, 'out': out_trace, 'c': c_trace, 'h': h_trace, 
                  'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces
    
#############################################################################################################################

class gatesLSTM_layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gatesLSTM_layer, self).__init__()
        
        self.f_gate_w_x = nn.Linear(input_size, hidden_size)
        self.f_gate_w_h = nn.Linear(hidden_size, hidden_size)
        self.i_gate_w_x = nn.Linear(input_size, hidden_size)
        self.i_gate_w_h = nn.Linear(hidden_size, hidden_size)
        self.o_gate_w_x = nn.Linear(input_size, hidden_size)
        self.o_gate_w_h = nn.Linear(hidden_size, hidden_size)
        
        self.input_w_x = nn.Linear(input_size, hidden_size)
        self.input_w_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, h_0, c_0):
        f_gate = torch.sigmoid(self.f_gate_w_x(x) + self.f_gate_w_h(h_0))
        i_gate = torch.sigmoid(self.i_gate_w_x(x) + self.i_gate_w_h(h_0))
        o_gate = torch.sigmoid(self.o_gate_w_x(x) + self.o_gate_w_h(h_0))
        
        candidate_c = torch.tanh(self.input_w_x(x) + self.input_w_h(h_0))
        
        c_t = torch.multiply(f_gate, c_0) + torch.multiply(i_gate, candidate_c)
        
        h_t = torch.multiply(o_gate, torch.tanh(c_t))
        out = h_t
        
        return out, h_t, c_t, i_gate, f_gate, o_gate

class pop_gatesLSTM(nn.Module):
    def __init__(self, params):
        super(pop_gatesLSTM, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        # subLSTM layer (input-to-hidden)
        self.gates_lstm = gatesLSTM_layer(self.input_size, self.hidden_size)
        self.l1 = nn.Linear(4*self.hidden_size, self.output_size) # linear layer (hidden-to-output)

        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, c_0, stage):
        # forward pass of data through network
        # input x dimensions -> ()

        out, h_n, c_n, i, f, o = self.gates_lstm(x, h_0, c_0)

        gates_out = torch.cat((out, i, f, o), axis=1)
        out = torch.sigmoid(self.l1(gates_out))
            
        return out, h_n, c_n
    
    def forward_loop(self, inputs, target_data, stage):
        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device)
        c_n = torch.zeros((1, self.hidden_size), device=device)
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
                c_n = torch.zeros((1, self.hidden_size), device=device)
        
            input_t = inputs[:, :, t].view(-1, self.input_size)
            
            # forward
            output, h_n, c_n = self.forward(input_t, h_n, c_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
        
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.gates_lstm.f_gate_w_x.weight.norm(p=1) + self.gates_lstm.f_gate_w_h.weight.norm(p=1) + \
                           self.gates_lstm.i_gate_w_x.weight.norm(p=1) + self.gates_lstm.i_gate_w_h.weight.norm(p=1) + \
                           self.gates_lstm.o_gate_w_x.weight.norm(p=1) + self.gates_lstm.o_gate_w_h.weight.norm(p=1) + \
                           self.gates_lstm.input_w_x.weight.norm(p=1) + self.gates_lstm.input_w_h.weight.norm(p=1) + \
                           self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):        
        model_info = {
             'w_gateslstm_x_f': self.gates_lstm.f_gate_w_x.weight, # weights - gatesLSTM layer, input-to-f gate
             'b_gateslstm_x_f': self.gates_lstm.f_gate_w_x.bias, # biases - gatesLSTM layer, input-to-f gate
             'w_gateslstm_h_f': self.gates_lstm.f_gate_w_h.weight, # weights - gatesLSTM layer, hidden-to-f gate
             'b_gateslstm_h_f': self.gates_lstm.f_gate_w_h.bias, # biases - gatesLSTM layer, hidden-to-f gate
             'w_gateslstm_x_i': self.gates_lstm.i_gate_w_x.weight, # weights - gatesLSTM layer, input-to-i gate
             'b_gateslstm_x_i': self.gates_lstm.i_gate_w_x.bias, # biases - gatesLSTM layer, input-to-i gate
             'w_gateslstm_h_i': self.gates_lstm.i_gate_w_h.weight, # weights - gatesLSTM layer, hidden-to-i gate
             'b_gateslstm_h_i': self.gates_lstm.i_gate_w_h.bias, # biases - gatesLSTM layer, hidden-to-i gate
             'w_gateslstm_x_o': self.gates_lstm.o_gate_w_x.weight, # weights - gatesLSTM layer, input-to-o gate
             'b_gateslstm_x_o': self.gates_lstm.o_gate_w_x.bias, # biases - gatesLSTM layer, input-to-o gate
             'w_gateslstm_h_o': self.gates_lstm.o_gate_w_h.weight, # weights - gatesLSTM layer, hidden-to-o gate
             'b_gateslstm_h_o': self.gates_lstm.o_gate_w_h.bias, # biases - gatesLSTM layer, hidden-to-o gate
             'w_gateslstm_x_in': self.gates_lstm.input_w_x.weight, # weights - gatesLSTM layer, input-to-hidden
             'b_gateslstm_x_in': self.gates_lstm.input_w_x.bias, # biases - gatesLSTM layer, input-to-hidden
             'w_gateslstm_h_in': self.gates_lstm.input_w_h.weight, # weights - gatesLSTM layer, hidden-to-hidden
             'b_gateslstm_h_in': self.gates_lstm.input_w_h.bias, # biases - gatesLSTM layer, hidden-to-hidden
             'w_l1': self.l1.weight, # weights - linear layer
             'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload subLSTM layer
        self.gates_lstm = gatesLSTM_layer(self.input_size, self.hidden_size)
        self.gates_lstm.f_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_gateslstm_x_f']).to(torch.float32).to(device))
        self.gates_lstm.f_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_gateslstm_x_f']).to(torch.float32).to(device))
        self.gates_lstm.f_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_gateslstm_h_f']).to(torch.float32).to(device))
        self.gates_lstm.f_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_gateslstm_h_f']).to(torch.float32).to(device))
        self.gates_lstm.i_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_gateslstm_x_i']).to(torch.float32).to(device))
        self.gates_lstm.i_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_gateslstm_x_i']).to(torch.float32).to(device))
        self.gates_lstm.i_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_gateslstm_h_i']).to(torch.float32).to(device))
        self.gates_lstm.i_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_gateslstm_h_i']).to(torch.float32).to(device))
        self.gates_lstm.o_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_gateslstm_x_o']).to(torch.float32).to(device))
        self.gates_lstm.o_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_gateslstm_x_o']).to(torch.float32).to(device))
        self.gates_lstm.o_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_gateslstm_h_o']).to(torch.float32).to(device))
        self.gates_lstm.o_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_gateslstm_h_o']).to(torch.float32).to(device))
        self.gates_lstm.input_w_x.weight = nn.Parameter(torch.tensor(info['w_gateslstm_x_in']).to(torch.float32).to(device))
        self.gates_lstm.input_w_x.bias = nn.Parameter(torch.tensor(info['b_gateslstm_x_in']).to(torch.float32).to(device))
        self.gates_lstm.input_w_h.weight = nn.Parameter(torch.tensor(info['w_gateslstm_h_in']).to(torch.float32).to(device))
        self.gates_lstm.input_w_h.bias = nn.Parameter(torch.tensor(info['b_gateslstm_h_in']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x, h_t, c_t):
        w_gateslstm_x_f = self.gates_lstm.f_gate_w_x.weight.detach().cpu().numpy()
        b_gateslstm_x_f = self.gates_lstm.f_gate_w_x.bias.detach().cpu().numpy()
        w_gateslstm_h_f = self.gates_lstm.f_gate_w_h.weight.detach().cpu().numpy()
        b_gateslstm_h_f = self.gates_lstm.f_gate_w_h.bias.detach().cpu().numpy()
        w_gateslstm_x_i = self.gates_lstm.i_gate_w_x.weight.detach().cpu().numpy()
        b_gateslstm_x_i = self.gates_lstm.i_gate_w_x.bias.detach().cpu().numpy()
        w_gateslstm_h_i = self.gates_lstm.i_gate_w_h.weight.detach().cpu().numpy()
        b_gateslstm_h_i = self.gates_lstm.i_gate_w_h.bias.detach().cpu().numpy()
        w_gateslstm_x_o = self.gates_lstm.o_gate_w_x.weight.detach().cpu().numpy()
        b_gateslstm_x_o = self.gates_lstm.o_gate_w_x.bias.detach().cpu().numpy()
        w_gateslstm_h_o = self.gates_lstm.o_gate_w_h.weight.detach().cpu().numpy()
        b_gateslstm_h_o = self.gates_lstm.o_gate_w_h.bias.detach().cpu().numpy()
        w_gateslstm_x_in = self.gates_lstm.input_w_x.weight.detach().cpu().numpy()
        b_gateslstm_x_in = self.gates_lstm.input_w_x.bias.detach().cpu().numpy()
        w_gateslstm_h_in = self.gates_lstm.input_w_h.weight.detach().cpu().numpy()
        b_gateslstm_h_in = self.gates_lstm.input_w_h.bias.detach().cpu().numpy()
        
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        
        i_t_pre_act = np.dot(w_gateslstm_x_i, x) + b_gateslstm_x_i[:, None] + np.dot(w_gateslstm_h_i, h_t) + \
            b_gateslstm_h_i[:, None]
        i_t = 1/(1 + np.exp(-i_t_pre_act))
        
        f_t_pre_act = np.dot(w_gateslstm_x_f, x) + b_gateslstm_x_f[:, None] + np.dot(w_gateslstm_h_f, h_t) + \
            b_gateslstm_h_f[:, None]
        f_t = 1/(1 + np.exp(-f_t_pre_act))
        
        out_t_pre_act = np.dot(w_gateslstm_x_o, x) + b_gateslstm_x_o[:, None] + np.dot(w_gateslstm_h_o, h_t) + \
            b_gateslstm_h_o[:, None]
        out_t = 1/(1 + np.exp(-out_t_pre_act))
        
        g_t_pre_act = np.dot(w_gateslstm_x_in, x) + b_gateslstm_x_in[:, None] + np.dot(w_gateslstm_h_in, h_t) + \
            b_gateslstm_h_in[:, None]
        g_t = np.tanh(g_t_pre_act)
        
        c_t = np.multiply(f_t, c_t) + np.multiply(i_t, g_t)
        
        h_t = np.multiply(out_t, np.tanh(c_t))
        
        gates_out = np.concatenate((h_t, i_t, f_t, out_t))
        o_t_pre_act = np.dot(w_l1, gates_out) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
        
        return o_t, o_t_pre_act, h_t, c_t, out_t, out_t_pre_act, g_t, g_t_pre_act, f_t, f_t_pre_act, i_t, i_t_pre_act    
    
    def np_forward_loop(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, c, out, out_pre_act, g, g_pre_act, f, f_pre_act, i, i_pre_act = self.np_forward(input_t, h, c)

            if t == 0:
                i_pre_act_trace = np.array(i_pre_act)
                i_trace = np.array(i)
                f_pre_act_trace = np.array(f_pre_act)
                f_trace = np.array(f)
                g_pre_act_trace = np.array(g_pre_act)
                g_trace = np.array(g)
                out_pre_act_trace = np.array(out_pre_act)
                out_trace = np.array(out)
                c_trace = np.array(c)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                i_pre_act_trace = np.append(i_pre_act_trace, i_pre_act, axis=1)
                i_trace = np.append(i_trace, i, axis=1)
                f_pre_act_trace = np.append(f_pre_act_trace, f_pre_act, axis=1)
                f_trace = np.append(f_trace, f, axis=1)
                g_pre_act_trace = np.append(g_pre_act_trace, g_pre_act, axis=1)
                g_trace = np.append(g_trace, g, axis=1)
                out_pre_act_trace = np.append(out_pre_act_trace, out_pre_act, axis=1)
                out_trace = np.append(out_trace, out, axis=1)
                c_trace = np.append(c_trace, c, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'i_pre_act': i_pre_act_trace, 'i': i_trace, 'f_pre_act': f_pre_act_trace, 'f': f_trace,
                  'g_pre_act': g_pre_act_trace, 'g': g_trace, 'out_pre_act': out_pre_act_trace, 'out': out_trace,
                  'c': c_trace, 'h': h_trace, 'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces
        
#############################################################################################################################

class gatesGRU_layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gatesGRU_layer, self).__init__()
        
        self.r_gate_w_x = nn.Linear(input_size, hidden_size)
        self.r_gate_w_h = nn.Linear(hidden_size, hidden_size)
        self.z_gate_w_x = nn.Linear(input_size, hidden_size)
        self.z_gate_w_h = nn.Linear(hidden_size, hidden_size)
        
        self.input_w_x = nn.Linear(input_size, hidden_size)
        self.input_w_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, h_0):
        r_gate = torch.sigmoid(self.r_gate_w_x(x) + self.r_gate_w_h(h_0))
        z_gate = torch.sigmoid(self.z_gate_w_x(x) + self.z_gate_w_h(h_0))
        
        candidate_h = torch.tanh(self.input_w_x(x) + torch.multiply(r_gate, self.input_w_h(h_0)))
        
        h_t = torch.multiply((1 - z_gate), candidate_h) + torch.multiply(z_gate, h_0)
        out = h_t
        
        return out, h_t, r_gate, z_gate

class pop_gatesGRU(nn.Module):
    def __init__(self, params):
        super(pop_gatesGRU, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        # subLSTM layer (input-to-hidden)
        self.gates_gru = gatesGRU_layer(self.input_size, self.hidden_size)
        self.l1 = nn.Linear(3*self.hidden_size, self.output_size) # linear layer (hidden-to-output)

        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, stage):
        # forward pass of data through network
        # input x dimensions -> ()

        out, h_n, r, z = self.gates_gru(x, h_0)

        gates_out = torch.cat((out, r, z), axis=1)
        out = torch.sigmoid(self.l1(gates_out))
            
        return out, h_n
    
    def forward_loop(self, inputs, target_data, stage):
        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device)
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
        
            input_t = inputs[:, :, t].view(-1, self.input_size)
            
            # forward
            output, h_n = self.forward(input_t, h_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
        
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.gates_gru.r_gate_w_x.weight.norm(p=1) + self.gates_gru.r_gate_w_h.weight.norm(p=1) + \
                           self.gates_gru.z_gate_w_x.weight.norm(p=1) + self.gates_gru.z_gate_w_h.weight.norm(p=1) + \
                           self.gates_gru.input_w_x.weight.norm(p=1) + self.gates_gru.input_w_h.weight.norm(p=1) + \
                           self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):        
        model_info = {
             'w_gatesgru_x_r': self.gates_gru.r_gate_w_x.weight, # weights - gatesGRU layer, input-to-r gate
             'b_gatesgru_x_r': self.gates_gru.r_gate_w_x.bias, # biases - gatesGRU layer, input-to-r gate
             'w_gatesgru_h_r': self.gates_gru.r_gate_w_h.weight, # weights - gatesGRU layer, hidden-to-r gate
             'b_gatesgru_h_r': self.gates_gru.r_gate_w_h.bias, # biases - gatesGRU layer, hidden-to-r gate
             'w_gatesgru_x_z': self.gates_gru.z_gate_w_x.weight, # weights - gatesGRU layer, input-to-z gate
             'b_gatesgru_x_z': self.gates_gru.z_gate_w_x.bias, # biases - gatesGRU layer, input-to-z gate
             'w_gatesgru_h_z': self.gates_gru.z_gate_w_h.weight, # weights - gatesGRU layer, hidden-to-z gate
             'b_gatesgru_h_z': self.gates_gru.z_gate_w_h.bias, # biases - gatesGRU layer, hidden-to-z gate
             'w_gatesgru_x_in': self.gates_gru.input_w_x.weight, # weights - gatesGRU layer, input-to-hidden
             'b_gatesgru_x_in': self.gates_gru.input_w_x.bias, # biases - gatesGRU layer, input-to-hidden
             'w_gatesgru_h_in': self.gates_gru.input_w_h.weight, # weights - gatesGRU layer, hidden-to-hidden
             'b_gatesgru_h_in': self.gates_gru.input_w_h.bias, # biases - gatesGRU layer, hidden-to-hidden
             'w_l1': self.l1.weight, # weights - linear layer
             'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload subLSTM layer
        self.gates_gru = gatesGRU_layer(self.input_size, self.hidden_size)
        self.gates_gru.r_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_gatesgru_x_r']).to(torch.float32).to(device))
        self.gates_gru.r_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_gatesgru_x_r']).to(torch.float32).to(device))
        self.gates_gru.r_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_gatesgru_h_r']).to(torch.float32).to(device))
        self.gates_gru.r_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_gatesgru_h_r']).to(torch.float32).to(device))
        self.gates_gru.z_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_gatesgru_x_z']).to(torch.float32).to(device))
        self.gates_gru.z_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_gatesgru_x_z']).to(torch.float32).to(device))
        self.gates_gru.z_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_gatesgru_h_z']).to(torch.float32).to(device))
        self.gates_gru.z_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_gatesgru_h_z']).to(torch.float32).to(device))
        self.gates_gru.input_w_x.weight = nn.Parameter(torch.tensor(info['w_gatesgru_x_in']).to(torch.float32).to(device))
        self.gates_gru.input_w_x.bias = nn.Parameter(torch.tensor(info['b_gatesgru_x_in']).to(torch.float32).to(device))
        self.gates_gru.input_w_h.weight = nn.Parameter(torch.tensor(info['w_gatesgru_h_in']).to(torch.float32).to(device))
        self.gates_gru.input_w_h.bias = nn.Parameter(torch.tensor(info['b_gatesgru_h_in']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x, h_t):
        w_gatesgru_x_r = self.gates_gru.r_gate_w_x.weight.detach().cpu().numpy()
        b_gatesgru_x_r = self.gates_gru.r_gate_w_x.bias.detach().cpu().numpy()
        w_gatesgru_h_r = self.gates_gru.r_gate_w_h.weight.detach().cpu().numpy()
        b_gatesgru_h_r = self.gates_gru.r_gate_w_h.bias.detach().cpu().numpy()
        w_gatesgru_x_z = self.gates_gru.z_gate_w_x.weight.detach().cpu().numpy()
        b_gatesgru_x_z = self.gates_gru.z_gate_w_x.bias.detach().cpu().numpy()
        w_gatesgru_h_z = self.gates_gru.z_gate_w_h.weight.detach().cpu().numpy()
        b_gatesgru_h_z = self.gates_gru.z_gate_w_h.bias.detach().cpu().numpy()
        w_gatesgru_x_in = self.gates_gru.input_w_x.weight.detach().cpu().numpy()
        b_gatesgru_x_in = self.gates_gru.input_w_x.bias.detach().cpu().numpy()
        w_gatesgru_h_in = self.gates_gru.input_w_h.weight.detach().cpu().numpy()
        b_gatesgru_h_in = self.gates_gru.input_w_h.bias.detach().cpu().numpy()
        
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        
        r_t_pre_act = np.dot(w_gatesgru_x_r, x) + b_gatesgru_x_r[:, None] + np.dot(w_gatesgru_h_r, h_t) + \
            b_gatesgru_h_r[:, None]
        r_t = 1/(1 + np.exp(-r_t_pre_act))
        
        z_t_pre_act = np.dot(w_gatesgru_x_z, x) + b_gatesgru_x_z[:, None] + np.dot(w_gatesgru_h_z, h_t) + \
            b_gatesgru_h_z[:, None]
        z_t = 1/(1 + np.exp(-z_t_pre_act))
        
        n_t_pre_act = np.dot(w_gatesgru_x_in, x) + b_gatesgru_x_in[:, None] + np.multiply(r_t, \
            np.dot(w_gatesgru_h_in, h_t) + b_gatesgru_h_in[:, None])
        n_t = np.tanh(n_t_pre_act)
        
        h_t = np.multiply((1 - z_t), n_t) + np.multiply(z_t, h_t)
        
        gates_out = np.concatenate((h_t, r_t, z_t))
        o_t_pre_act = np.dot(w_l1, gates_out) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
        
        return o_t, o_t_pre_act, h_t, n_t, n_t_pre_act, z_t, z_t_pre_act, r_t, r_t_pre_act    
    
    def np_forward_loop(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1))

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, n, n_pre_act, z, z_pre_act, r, r_pre_act = self.np_forward(input_t, h)

            if t == 0:
                r_pre_act_trace = np.array(r_pre_act)
                r_trace = np.array(r)
                z_pre_act_trace = np.array(z_pre_act)
                z_trace = np.array(z)
                n_pre_act_trace = np.array(n_pre_act)
                n_trace = np.array(n)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                r_pre_act_trace = np.append(r_pre_act_trace, r_pre_act, axis=1)
                r_trace = np.append(r_trace, r, axis=1)
                z_pre_act_trace = np.append(z_pre_act_trace, z_pre_act, axis=1)
                z_trace = np.append(z_trace, z, axis=1)
                n_pre_act_trace = np.append(n_pre_act_trace, n_pre_act, axis=1)
                n_trace = np.append(n_trace, n, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'r_pre_act': r_pre_act_trace, 'r': r_trace, 'z_pre_act': z_pre_act_trace, 'z': z_trace,
                  'n_pre_act': n_pre_act_trace, 'n': n_trace, 'h': h_trace, 'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces
    
#############################################################################################################################

class gatessubLSTM_layer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gatessubLSTM_layer, self).__init__()
        
        self.f_gate_w_x = nn.Linear(input_size, hidden_size)
        self.f_gate_w_h = nn.Linear(hidden_size, hidden_size)
        self.i_gate_w_x = nn.Linear(input_size, hidden_size)
        self.i_gate_w_h = nn.Linear(hidden_size, hidden_size)
        self.o_gate_w_x = nn.Linear(input_size, hidden_size)
        self.o_gate_w_h = nn.Linear(hidden_size, hidden_size)
        
        self.input_w_x = nn.Linear(input_size, hidden_size)
        self.input_w_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, h_0, c_0):
        f_gate = torch.sigmoid(self.f_gate_w_x(x) + self.f_gate_w_h(h_0))
        i_gate = torch.sigmoid(self.i_gate_w_x(x) + self.i_gate_w_h(h_0))
        o_gate = torch.sigmoid(self.o_gate_w_x(x) + self.o_gate_w_h(h_0))
        
        candidate_c = torch.sigmoid(self.input_w_x(x) + self.input_w_h(h_0))
        
        c_t = torch.multiply(f_gate, c_0) + candidate_c - i_gate
        
        h_t = torch.sigmoid(c_t) - o_gate
        out = h_t
        
        return out, h_t, c_t, i_gate, f_gate, o_gate

class pop_gatessubLSTM(nn.Module):
    def __init__(self, params):
        super(pop_gatessubLSTM, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        # subLSTM layer (input-to-hidden)
        self.gates_sublstm = gatessubLSTM_layer(self.input_size, self.hidden_size)
        self.l1 = nn.Linear(4*self.hidden_size, self.output_size) # linear layer (hidden-to-output)

        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, c_0, stage):
        # forward pass of data through network
        # input x dimensions -> ()

        out, h_n, c_n, i, f, o = self.gates_sublstm(x, h_0, c_0)

        gates_out = torch.cat((out, i, f, o), axis=1)
        out = torch.sigmoid(self.l1(gates_out))
            
        return out, h_n, c_n
    
    def forward_loop(self, inputs, target_data, stage):
        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device)
        c_n = torch.zeros((1, self.hidden_size), device=device)
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
                c_n = torch.zeros((1, self.hidden_size), device=device)
        
            input_t = inputs[:, :, t].view(-1, self.input_size)
            
            # forward
            output, h_n, c_n = self.forward(input_t, h_n, c_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
        
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.gates_sublstm.f_gate_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.f_gate_w_h.weight.norm(p=1) + \
                           self.gates_sublstm.i_gate_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.i_gate_w_h.weight.norm(p=1) + \
                           self.gates_sublstm.o_gate_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.o_gate_w_h.weight.norm(p=1) + \
                           self.gates_sublstm.input_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.input_w_h.weight.norm(p=1) + \
                           self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):        
        model_info = {
             'w_gatessublstm_x_f': self.gates_sublstm.f_gate_w_x.weight, # weights - gatesLSTM layer, input-to-f gate
             'b_gatessublstm_x_f': self.gates_sublstm.f_gate_w_x.bias, # biases - gatesLSTM layer, input-to-f gate
             'w_gatessublstm_h_f': self.gates_sublstm.f_gate_w_h.weight, # weights - gatesLSTM layer, hidden-to-f gate
             'b_gatessublstm_h_f': self.gates_sublstm.f_gate_w_h.bias, # biases - gatesLSTM layer, hidden-to-f gate
             'w_gatessublstm_x_i': self.gates_sublstm.i_gate_w_x.weight, # weights - gatesLSTM layer, input-to-i gate
             'b_gatessublstm_x_i': self.gates_sublstm.i_gate_w_x.bias, # biases - gatesLSTM layer, input-to-i gate
             'w_gatessublstm_h_i': self.gates_sublstm.i_gate_w_h.weight, # weights - gatesLSTM layer, hidden-to-i gate
             'b_gatessublstm_h_i': self.gates_sublstm.i_gate_w_h.bias, # biases - gatesLSTM layer, hidden-to-i gate
             'w_gatessublstm_x_o': self.gates_sublstm.o_gate_w_x.weight, # weights - gatesLSTM layer, input-to-o gate
             'b_gatessublstm_x_o': self.gates_sublstm.o_gate_w_x.bias, # biases - gatesLSTM layer, input-to-o gate
             'w_gatessublstm_h_o': self.gates_sublstm.o_gate_w_h.weight, # weights - gatesLSTM layer, hidden-to-o gate
             'b_gatessublstm_h_o': self.gates_sublstm.o_gate_w_h.bias, # biases - gatesLSTM layer, hidden-to-o gate
             'w_gatessublstm_x_in': self.gates_sublstm.input_w_x.weight, # weights - gatesLSTM layer, input-to-hidden
             'b_gatessublstm_x_in': self.gates_sublstm.input_w_x.bias, # biases - gatesLSTM layer, input-to-hidden
             'w_gatessublstm_h_in': self.gates_sublstm.input_w_h.weight, # weights - gatesLSTM layer, hidden-to-hidden
             'b_gatessublstm_h_in': self.gates_sublstm.input_w_h.bias, # biases - gatesLSTM layer, hidden-to-hidden
             'w_l1': self.l1.weight, # weights - linear layer
             'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload subLSTM layer
        self.gates_sublstm = gatessubLSTM_layer(self.input_size, self.hidden_size)
        self.gates_sublstm.f_gate_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_f']).to(torch.float32).to(device))
        self.gates_sublstm.f_gate_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_f']).to(torch.float32).to(device))
        self.gates_sublstm.f_gate_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_f']).to(torch.float32).to(device))
        self.gates_sublstm.f_gate_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_f']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_i']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_i']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_i']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_i']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_o']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_o']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_o']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_o']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_in']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_in']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_in']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_in']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x, h_t, c_t):
        w_gatessublstm_x_f = self.gates_sublstm.f_gate_w_x.weight.detach().cpu().numpy()
        b_gatessublstm_x_f = self.gates_sublstm.f_gate_w_x.bias.detach().cpu().numpy()
        w_gatessublstm_h_f = self.gates_sublstm.f_gate_w_h.weight.detach().cpu().numpy()
        b_gatessublstm_h_f = self.gates_sublstm.f_gate_w_h.bias.detach().cpu().numpy()
        w_gatessublstm_x_i = self.gates_sublstm.i_gate_w_x.weight.detach().cpu().numpy()
        b_gatessublstm_x_i = self.gates_sublstm.i_gate_w_x.bias.detach().cpu().numpy()
        w_gatessublstm_h_i = self.gates_sublstm.i_gate_w_h.weight.detach().cpu().numpy()
        b_gatessublstm_h_i = self.gates_sublstm.i_gate_w_h.bias.detach().cpu().numpy()
        w_gatessublstm_x_o = self.gates_sublstm.o_gate_w_x.weight.detach().cpu().numpy()
        b_gatessublstm_x_o = self.gates_sublstm.o_gate_w_x.bias.detach().cpu().numpy()
        w_gatessublstm_h_o = self.gates_sublstm.o_gate_w_h.weight.detach().cpu().numpy()
        b_gatessublstm_h_o = self.gates_sublstm.o_gate_w_h.bias.detach().cpu().numpy()
        w_gatessublstm_x_in = self.gates_sublstm.input_w_x.weight.detach().cpu().numpy()
        b_gatessublstm_x_in = self.gates_sublstm.input_w_x.bias.detach().cpu().numpy()
        w_gatessublstm_h_in = self.gates_sublstm.input_w_h.weight.detach().cpu().numpy()
        b_gatessublstm_h_in = self.gates_sublstm.input_w_h.bias.detach().cpu().numpy()
        
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        
        i_t_pre_act = np.dot(w_gatessublstm_x_i, x) + b_gatessublstm_x_i[:, None] + np.dot(w_gatessublstm_h_i, h_t) + \
            b_gatessublstm_h_i[:, None]
        i_t = 1/(1 + np.exp(-i_t_pre_act))
        
        f_t_pre_act = np.dot(w_gatessublstm_x_f, x) + b_gatessublstm_x_f[:, None] + np.dot(w_gatessublstm_h_f, h_t) + \
            b_gatessublstm_h_f[:, None]
        f_t = 1/(1 + np.exp(-f_t_pre_act))
        
        out_t_pre_act = np.dot(w_gatessublstm_x_o, x) + b_gatessublstm_x_o[:, None] + np.dot(w_gatessublstm_h_o, h_t) + \
            b_gatessublstm_h_o[:, None]
        out_t = 1/(1 + np.exp(-out_t_pre_act))
        
        g_t_pre_act = np.dot(w_gatessublstm_x_in, x) + b_gatessublstm_x_in[:, None] + np.dot(w_gatessublstm_h_in, h_t) + \
            b_gatessublstm_h_in[:, None]
        g_t = 1/(1 + np.exp(-g_t_pre_act))
        
        c_t = np.multiply(f_t, c_t) + g_t - i_t
        
        h_t = 1/(1 + np.exp(-c_t)) - out_t
        
        gates_out = np.concatenate((h_t, i_t, f_t, out_t))
        o_t_pre_act = np.dot(w_l1, gates_out) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))

        return o_t, o_t_pre_act, h_t, c_t, out_t, out_t_pre_act, g_t, g_t_pre_act, f_t, f_t_pre_act, i_t, i_t_pre_act    
    
    def np_forward_loop(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, c, out, out_pre_act, g, g_pre_act, f, f_pre_act, i, i_pre_act = self.np_forward(input_t, h, c)

            if t == 0:
                i_pre_act_trace = np.array(i_pre_act)
                i_trace = np.array(i)
                f_pre_act_trace = np.array(f_pre_act)
                f_trace = np.array(f)
                g_pre_act_trace = np.array(g_pre_act)
                g_trace = np.array(g)
                out_pre_act_trace = np.array(out_pre_act)
                out_trace = np.array(out)
                c_trace = np.array(c)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                i_pre_act_trace = np.append(i_pre_act_trace, i_pre_act, axis=1)
                i_trace = np.append(i_trace, i, axis=1)
                f_pre_act_trace = np.append(f_pre_act_trace, f_pre_act, axis=1)
                f_trace = np.append(f_trace, f, axis=1)
                g_pre_act_trace = np.append(g_pre_act_trace, g_pre_act, axis=1)
                g_trace = np.append(g_trace, g, axis=1)
                out_pre_act_trace = np.append(out_pre_act_trace, out_pre_act, axis=1)
                out_trace = np.append(out_trace, out, axis=1)
                c_trace = np.append(c_trace, c, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'i_pre_act': i_pre_act_trace, 'i': i_trace, 'f_pre_act': f_pre_act_trace, 'f': f_trace,
                  'g_pre_act': g_pre_act_trace, 'g': g_trace, 'out_pre_act': out_pre_act_trace, 'out': out_trace,
                  'c': c_trace, 'h': h_trace, 'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces

#############################################################################################################################

class pop_gatesfLSTM(nn.Module):
    def __init__(self, params):
        super(pop_gatesfLSTM, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        self.t_gate_w_x = nn.Linear(self.input_size, self.input_size)
        self.t_gate_w_h = nn.Linear(self.hidden_size, self.input_size)
        self.gates_sublstm = gatessubLSTM_layer(self.input_size, self.hidden_size)
        self.l1 = nn.Linear(5*self.hidden_size, self.output_size) # linear layer (hidden-to-output)

        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        if 't_gate_memory' in params:    
            self.t_gate_memory = params['t_gate_memory']
        else:
            self.t_gate_memory = 'hidden'
        
        if 'split' in params:
            self.split = params['split']
        else:
            self.split = 1
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, c_0, stage):
        if self.t_gate_memory == 'hidden':
            t_gate = torch.sigmoid(self.t_gate_w_x(x) + self.t_gate_w_h(h_0))
        elif self.t_gate_memory == 'cell':
            t_gate = torch.sigmoid(self.t_gate_w_x(x) + self.t_gate_w_h(c_0))
            
        x_t_gate = torch.multiply(t_gate, x)
        out, h_n, c_n, i, f, o = self.gates_sublstm(x_t_gate, h_0, c_0)
        
        gates_out = torch.cat((out, i, f, o, c_n), axis=1)
        out = torch.sigmoid(self.l1(gates_out))
            
        return out, h_n, c_n
    
    def forward_loop(self, inputs, target_data, stage):
        if stage == 'train':
            stim_size = self.stim_size
        elif stage == 'val' or stage == 'test':
            n_data = int(np.floor(inputs.size(2)/(self.stim_size/self.split)))
            stim_size = int(inputs.size(2)/n_data)

        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device)
        c_n = torch.zeros((1, self.hidden_size), device=device)
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            #print(t)
            if t % stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
                c_n = torch.zeros((1, self.hidden_size), device=device)
            
            input_t = inputs[:, :, t].view(-1, self.input_size)
            
            # forward
            output, h_n, c_n = self.forward(input_t, h_n, c_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
            
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.t_gate_w_x.weight.norm(p=1) + self.t_gate_w_h.weight.norm(p=1) + \
                           self.gates_sublstm.f_gate_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.f_gate_w_h.weight.norm(p=1) + \
                           self.gates_sublstm.i_gate_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.i_gate_w_h.weight.norm(p=1) + \
                           self.gates_sublstm.o_gate_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.o_gate_w_h.weight.norm(p=1) + \
                           self.gates_sublstm.input_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.input_w_h.weight.norm(p=1) + self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):        
        model_info = {
            'w_x_t': self.t_gate_w_x.weight, # weights - input-to-t gate
            'b_x_t': self.t_gate_w_x.bias, # biases - input-to-t gate
            'w_h_t': self.t_gate_w_h.weight, # weights - hidden-to-t gate
            'b_h_t': self.t_gate_w_h.bias, # biases - hidden-to-t gate
            'w_gatessublstm_x_f': self.gates_sublstm.f_gate_w_x.weight, # weights - subLSTM layer, input-to-f gate
            'b_gatessublstm_x_f': self.gates_sublstm.f_gate_w_x.bias, # biases - subLSTM layer, input-to-f gate
            'w_gatessublstm_h_f': self.gates_sublstm.f_gate_w_h.weight, # weights - subLSTM layer, hidden-to-f gate
            'b_gatessublstm_h_f': self.gates_sublstm.f_gate_w_h.bias, # biases - subLSTM layer, hidden-to-f gate
            'w_gatessublstm_x_i': self.gates_sublstm.i_gate_w_x.weight, # weights - subLSTM layer, input-to-i gate
            'b_gatessublstm_x_i': self.gates_sublstm.i_gate_w_x.bias, # biases - subLSTM layer, input-to-i gate
            'w_gatessublstm_h_i': self.gates_sublstm.i_gate_w_h.weight, # weights - subLSTM layer, hidden-to-i gate
            'b_gatessublstm_h_i': self.gates_sublstm.i_gate_w_h.bias, # biases - subLSTM layer, hidden-to-i gate
            'w_gatessublstm_x_o': self.gates_sublstm.o_gate_w_x.weight, # weights - subLSTM layer, input-to-o gate
            'b_gatessublstm_x_o': self.gates_sublstm.o_gate_w_x.bias, # biases - subLSTM layer, input-to-o gate
            'w_gatessublstm_h_o': self.gates_sublstm.o_gate_w_h.weight, # weights - subLSTM layer, hidden-to-o gate
            'b_gatessublstm_h_o': self.gates_sublstm.o_gate_w_h.bias, # biases - subLSTM layer, hidden-to-o gate
            'w_gatessublstm_x_in': self.gates_sublstm.input_w_x.weight, # weights - subLSTM layer, input-to-hidden
            'b_gatessublstm_x_in': self.gates_sublstm.input_w_x.bias, # biases - subLSTM layer, input-to-hidden
            'w_gatessublstm_h_in': self.gates_sublstm.input_w_h.weight, # weights - subLSTM layer, hidden-to-hidden
            'b_gatessublstm_h_in': self.gates_sublstm.input_w_h.bias, # biases - subLSTM layer, hidden-to-hidden
            'w_l1': self.l1.weight, # weights - linear layer
            'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload t gate layer
        self.t_gate_w_x = nn.Linear(self.input_size, self.input_size)
        self.t_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_x_t']).to(torch.float32).to(device))
        self.t_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_x_t']).to(torch.float32).to(device))
        
        self.t_gate_w_h = nn.Linear(self.hidden_size, self.input_size)
        self.t_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_h_t']).to(torch.float32).to(device))
        self.t_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_h_t']).to(torch.float32).to(device))
        
        # reload subLSTM layer
        self.gates_sublstm = gatessubLSTM_layer(self.input_size, self.hidden_size)
        self.gates_sublstm.f_gate_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_f']).to(torch.float32).to(device))
        self.gates_sublstm.f_gate_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_f']).to(torch.float32).to(device))
        self.gates_sublstm.f_gate_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_f']).to(torch.float32).to(device))
        self.gates_sublstm.f_gate_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_f']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_i']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_i']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_i']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_i']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_o']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_o']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_o']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_o']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_in']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_in']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_in']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_in']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x, h_t, c_t, model_params):
        w_x_t = model_params['w_x_t']
        b_x_t = model_params['b_x_t']
        w_h_t = model_params['w_h_t']
        b_h_t = model_params['b_h_t']
        
        w_gatessublstm_x_f = model_params['w_gatessublstm_x_f']
        b_gatessublstm_x_f = model_params['b_gatessublstm_x_f']
        w_gatessublstm_h_f = model_params['w_gatessublstm_h_f']
        b_gatessublstm_h_f = model_params['b_gatessublstm_h_f']
        w_gatessublstm_x_i = model_params['w_gatessublstm_x_i']
        b_gatessublstm_x_i = model_params['b_gatessublstm_x_i']
        w_gatessublstm_h_i = model_params['w_gatessublstm_h_i']
        b_gatessublstm_h_i = model_params['b_gatessublstm_h_i']
        w_gatessublstm_x_o = model_params['w_gatessublstm_x_o']
        b_gatessublstm_x_o = model_params['b_gatessublstm_x_o']
        w_gatessublstm_h_o = model_params['w_gatessublstm_h_o']
        b_gatessublstm_h_o = model_params['b_gatessublstm_h_o']
        w_gatessublstm_x_in = model_params['w_gatessublstm_x_in']
        b_gatessublstm_x_in = model_params['b_gatessublstm_x_in']
        w_gatessublstm_h_in = model_params['w_gatessublstm_h_in']
        b_gatessublstm_h_in = model_params['b_gatessublstm_h_in']
        
        w_l1 = model_params['w_l1']
        b_l1 = model_params['b_l1']
        
        if self.t_gate_memory == 'hidden':
            t_t_pre_act = np.dot(w_x_t, x) + b_x_t[:, None] + np.dot(w_h_t, h_t) + b_h_t[:, None]
        elif self.t_gate_memory == 'cell':
            t_t_pre_act = np.dot(w_x_t, x) + b_x_t[:, None] + np.dot(w_h_t, c_t) + b_h_t[:, None]
        t_t = 1/(1 + np.exp(-t_t_pre_act))
        
        x_t_gate = np.multiply(t_t, x)
        
        i_t_pre_act = np.dot(w_gatessublstm_x_i, x_t_gate) + b_gatessublstm_x_i[:, None] + \
            np.dot(w_gatessublstm_h_i, h_t) + b_gatessublstm_h_i[:, None]
        i_t = 1/(1 + np.exp(-i_t_pre_act))
        
        f_t_pre_act = np.dot(w_gatessublstm_x_f, x_t_gate) + b_gatessublstm_x_f[:, None] + \
            np.dot(w_gatessublstm_h_f, h_t) + b_gatessublstm_h_f[:, None]
        f_t = 1/(1 + np.exp(-f_t_pre_act))
        
        out_t_pre_act = np.dot(w_gatessublstm_x_o, x_t_gate) + b_gatessublstm_x_o[:, None] + \
            np.dot(w_gatessublstm_h_o, h_t) + b_gatessublstm_h_o[:, None]
        out_t = 1/(1 + np.exp(-out_t_pre_act))

        g_t_pre_act = np.dot(w_gatessublstm_x_in, x_t_gate) + b_gatessublstm_x_in[:, None] + \
            np.dot(w_gatessublstm_h_in, h_t) + b_gatessublstm_h_in[:, None]
        g_t = 1/(1 + np.exp(-g_t_pre_act))
        
        c_t = np.multiply(f_t, c_t) + g_t - i_t
        
        h_t = 1/(1 + np.exp(-c_t)) - out_t
        
        gates_out = np.concatenate((h_t, i_t, f_t, out_t, c_t))
        o_t_pre_act = np.dot(w_l1, gates_out) + b_l1[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
            
        return o_t, o_t_pre_act, h_t, c_t, out_t, out_t_pre_act, g_t, g_t_pre_act, f_t, f_t_pre_act, i_t, i_t_pre_act, \
            x_t_gate, t_t, t_t_pre_act
    
    def np_forward_loop(self, inputs):
        np_model_params = {'w_x_t': self.t_gate_w_x.weight.detach().cpu().numpy(),
                           'b_x_t': self.t_gate_w_x.bias.detach().cpu().numpy(),
                           'w_h_t': self.t_gate_w_h.weight.detach().cpu().numpy(),
                           'b_h_t': self.t_gate_w_h.bias.detach().cpu().numpy(),
                           'w_gatessublstm_x_f': self.gates_sublstm.f_gate_w_x.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_x_f': self.gates_sublstm.f_gate_w_x.bias.detach().cpu().numpy(), 
                           'w_gatessublstm_h_f': self.gates_sublstm.f_gate_w_h.weight.detach().cpu().numpy(),
                           'b_gatessublstm_h_f': self.gates_sublstm.f_gate_w_h.bias.detach().cpu().numpy(), 
                           'w_gatessublstm_x_i': self.gates_sublstm.i_gate_w_x.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_x_i': self.gates_sublstm.i_gate_w_x.bias.detach().cpu().numpy(),
                           'w_gatessublstm_h_i': self.gates_sublstm.i_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_h_i': self.gates_sublstm.i_gate_w_h.bias.detach().cpu().numpy(), 
                           'w_gatessublstm_x_o': self.gates_sublstm.o_gate_w_x.weight.detach().cpu().numpy(),
                           'b_gatessublstm_x_o': self.gates_sublstm.o_gate_w_x.bias.detach().cpu().numpy(), 
                           'w_gatessublstm_h_o': self.gates_sublstm.o_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_h_o': self.gates_sublstm.o_gate_w_h.bias.detach().cpu().numpy(),
                           'w_gatessublstm_x_in': self.gates_sublstm.input_w_x.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_x_in': self.gates_sublstm.input_w_x.bias.detach().cpu().numpy(), 
                           'w_gatessublstm_h_in': self.gates_sublstm.input_w_h.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_h_in': self.gates_sublstm.input_w_h.bias.detach().cpu().numpy(), 
                           'w_l1': self.l1.weight.detach().cpu().numpy(), 'b_l1': self.l1.bias.detach().cpu().numpy()}
        
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            #print(t)
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, c, out, out_pre_act, g, g_pre_act, f, f_pre_act, i, i_pre_act, x_t_gated, t_gate, \
                t_gate_pre_act = self.np_forward(input_t, h, c, np_model_params)

            if t == 0:
                t_pre_act_trace = np.array(t_gate_pre_act)
                t_trace = np.array(t_gate)
                x_t_gated_trace = np.array(x_t_gated)
                i_pre_act_trace = np.array(i_pre_act)
                i_trace = np.array(i)
                f_pre_act_trace = np.array(f_pre_act)
                f_trace = np.array(f)
                g_pre_act_trace = np.array(g_pre_act)
                g_trace = np.array(g)
                out_pre_act_trace = np.array(out_pre_act)
                out_trace = np.array(out)
                c_trace = np.array(c)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                t_pre_act_trace = np.append(t_pre_act_trace, t_gate_pre_act, axis=1)
                t_trace = np.append(t_trace, t_gate, axis=1)
                x_t_gated_trace = np.append(x_t_gated_trace, x_t_gated, axis=1)
                i_pre_act_trace = np.append(i_pre_act_trace, i_pre_act, axis=1)
                i_trace = np.append(i_trace, i, axis=1)
                f_pre_act_trace = np.append(f_pre_act_trace, f_pre_act, axis=1)
                f_trace = np.append(f_trace, f, axis=1)
                g_pre_act_trace = np.append(g_pre_act_trace, g_pre_act, axis=1)
                g_trace = np.append(g_trace, g, axis=1)
                out_pre_act_trace = np.append(out_pre_act_trace, out_pre_act, axis=1)
                out_trace = np.append(out_trace, out, axis=1)
                c_trace = np.append(c_trace, c, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'t_pre_act': t_pre_act_trace, 't': t_trace, 'x_t_gated': x_t_gated_trace, 'i_pre_act': i_pre_act_trace, 
                  'i': i_trace, 'f_pre_act': f_pre_act_trace, 'f': f_trace, 'g_pre_act': g_pre_act_trace, 'g': g_trace, 
                  'out_pre_act': out_pre_act_trace, 'out': out_trace, 'c': c_trace, 'h': h_trace, 
                  'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces
    
#############################################################################################################################

class pop_directrcLSTM(nn.Module):
    def __init__(self, params):
        super(pop_directrcLSTM, self).__init__()
        
        self.input_size = params['input_size'] # input layer size
        self.added_layers_size = 256
        self.hidden_size = params['hidden_size'] # hidden layer size
        self.output_size = params['output_size'] # output layer size
        
        self.t_gate_w_x = nn.Linear(self.input_size, self.input_size)
        self.t_gate_w_h = nn.Linear(self.hidden_size, self.input_size)
        self.added_l1 = nn.Linear(self.input_size, self.added_layers_size)
        self.added_to_out = nn.Linear(self.added_layers_size, self.output_size)
        self.gates_sublstm = gatessubLSTM_layer(self.added_layers_size, self.hidden_size)
        self.l1 = nn.Linear(self.hidden_size, self.output_size) # linear layer (hidden-to-output)

        self.dropout = params['dropout']
        self.input_dropout = nn.Dropout(p=0.2) # fraction of units silenced, in dropout, in input layer
        self.hidden_dropout = nn.Dropout(p=0.5) # fraction of units silenced, in dropout, in hidden layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        
        if 't_gate_memory' in params:    
            self.t_gate_memory = params['t_gate_memory']
        else:
            self.t_gate_memory = 'hidden'
        
        if 'split' in params:
            self.split = params['split']
        else:
            self.split = 1
        
        self.initialisation = params['initialisation']
        
    def forward(self, x, h_0, c_0, stage):
        if self.t_gate_memory == 'hidden':
            t_gate = torch.sigmoid(self.t_gate_w_x(x) + self.t_gate_w_h(h_0))
        elif self.t_gate_memory == 'cell':
            t_gate = torch.sigmoid(self.t_gate_w_x(x) + self.t_gate_w_h(c_0))
            
        x_t_gated = torch.multiply(t_gate, x)
        x = torch.sigmoid(self.added_l1(x_t_gated))
        sublstm_out, h_n, c_n, i, f, o = self.gates_sublstm(x, h_0, c_0)
        out = torch.sigmoid(self.l1(sublstm_out) + self.added_to_out(x))
            
        return out, h_n, c_n
    
    def forward_loop(self, inputs, target_data, stage):
        if stage == 'train':
            stim_size = self.stim_size
        elif stage == 'val' or stage == 'test':
            n_data = int(np.floor(inputs.size(2)/(self.stim_size/self.split)))
            stim_size = int(inputs.size(2)/n_data)

        targets = torch.transpose(target_data, 0, 1)
        outputs = torch.tensor([], device=device)
        
        h_n = torch.zeros((1, self.hidden_size), device=device)
        c_n = torch.zeros((1, self.hidden_size), device=device)
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            #print(t)
            if t % stim_size == 0:
                h_n = torch.zeros((1, self.hidden_size), device=device)
                c_n = torch.zeros((1, self.hidden_size), device=device)
            
            input_t = inputs[:, :, t].view(-1, self.input_size)
            
            # forward
            output, h_n, c_n = self.forward(input_t, h_n, c_n, stage)
            outputs = torch.cat((outputs, output), dim=0)
            
        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.t_gate_w_x.weight.norm(p=1) + self.t_gate_w_h.weight.norm(p=1) + \
                           self.added_l1.weight.norm(p=1) + \
                           self.added_to_out.weight.norm(p=1) + \
                           self.gates_sublstm.f_gate_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.f_gate_w_h.weight.norm(p=1) + \
                           self.gates_sublstm.i_gate_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.i_gate_w_h.weight.norm(p=1) + \
                           self.gates_sublstm.o_gate_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.o_gate_w_h.weight.norm(p=1) + \
                           self.gates_sublstm.input_w_x.weight.norm(p=1) + \
                           self.gates_sublstm.input_w_h.weight.norm(p=1) + self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):        
        model_info = {
            'w_x_t': self.t_gate_w_x.weight, # weights - input-to-t gate
            'b_x_t': self.t_gate_w_x.bias, # biases - input-to-t gate
            'w_h_t': self.t_gate_w_h.weight, # weights - hidden-to-t gate
            'b_h_t': self.t_gate_w_h.bias, # biases - hidden-to-t gate
            'w_added1': self.added_l1.weight, # weights - added linear layer
            'b_added1': self.added_l1.bias, # biases - added linear layer
            'w_added_to_out': self.added_to_out.weight, # weights - added linear layer
            'b_added_to_out': self.added_to_out.bias, # biases - added linear layer
            'w_gatessublstm_x_f': self.gates_sublstm.f_gate_w_x.weight, # weights - subLSTM layer, input-to-f gate
            'b_gatessublstm_x_f': self.gates_sublstm.f_gate_w_x.bias, # biases - subLSTM layer, input-to-f gate
            'w_gatessublstm_h_f': self.gates_sublstm.f_gate_w_h.weight, # weights - subLSTM layer, hidden-to-f gate
            'b_gatessublstm_h_f': self.gates_sublstm.f_gate_w_h.bias, # biases - subLSTM layer, hidden-to-f gate
            'w_gatessublstm_x_i': self.gates_sublstm.i_gate_w_x.weight, # weights - subLSTM layer, input-to-i gate
            'b_gatessublstm_x_i': self.gates_sublstm.i_gate_w_x.bias, # biases - subLSTM layer, input-to-i gate
            'w_gatessublstm_h_i': self.gates_sublstm.i_gate_w_h.weight, # weights - subLSTM layer, hidden-to-i gate
            'b_gatessublstm_h_i': self.gates_sublstm.i_gate_w_h.bias, # biases - subLSTM layer, hidden-to-i gate
            'w_gatessublstm_x_o': self.gates_sublstm.o_gate_w_x.weight, # weights - subLSTM layer, input-to-o gate
            'b_gatessublstm_x_o': self.gates_sublstm.o_gate_w_x.bias, # biases - subLSTM layer, input-to-o gate
            'w_gatessublstm_h_o': self.gates_sublstm.o_gate_w_h.weight, # weights - subLSTM layer, hidden-to-o gate
            'b_gatessublstm_h_o': self.gates_sublstm.o_gate_w_h.bias, # biases - subLSTM layer, hidden-to-o gate
            'w_gatessublstm_x_in': self.gates_sublstm.input_w_x.weight, # weights - subLSTM layer, input-to-hidden
            'b_gatessublstm_x_in': self.gates_sublstm.input_w_x.bias, # biases - subLSTM layer, input-to-hidden
            'w_gatessublstm_h_in': self.gates_sublstm.input_w_h.weight, # weights - subLSTM layer, hidden-to-hidden
            'b_gatessublstm_h_in': self.gates_sublstm.input_w_h.bias, # biases - subLSTM layer, hidden-to-hidden
            'w_l1': self.l1.weight, # weights - linear layer
            'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload t gate layer
        self.t_gate_w_x = nn.Linear(self.input_size, self.input_size)
        self.t_gate_w_x.weight = nn.Parameter(torch.tensor(info['w_x_t']).to(torch.float32).to(device))
        self.t_gate_w_x.bias = nn.Parameter(torch.tensor(info['b_x_t']).to(torch.float32).to(device))
        
        self.t_gate_w_h = nn.Linear(self.hidden_size, self.input_size)
        self.t_gate_w_h.weight = nn.Parameter(torch.tensor(info['w_h_t']).to(torch.float32).to(device))
        self.t_gate_w_h.bias = nn.Parameter(torch.tensor(info['b_h_t']).to(torch.float32).to(device))
        
        # reload added linear layer
        self.added_l1 = nn.Linear(self.input_size, self.added_layers_size)
        self.added_l1.weight = nn.Parameter(torch.tensor(info['w_added1']).to(torch.float32).to(device))
        self.added_l1.bias = nn.Parameter(torch.tensor(info['b_added1']).to(torch.float32).to(device))
        
        # reload added-to-out layer
        self.added_to_out = nn.Linear(self.added_layers_size, self.output_size)
        self.added_to_out.weight = nn.Parameter(torch.tensor(info['w_added_to_out']).to(torch.float32).to(device))
        self.added_to_out.bias = nn.Parameter(torch.tensor(info['b_added_to_out']).to(torch.float32).to(device))
        
        # reload subLSTM layer
        self.gates_sublstm = gatessubLSTM_layer(self.input_size, self.hidden_size)
        self.gates_sublstm.f_gate_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_f']).to(torch.float32).to(device))
        self.gates_sublstm.f_gate_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_f']).to(torch.float32).to(device))
        self.gates_sublstm.f_gate_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_f']).to(torch.float32).to(device))
        self.gates_sublstm.f_gate_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_f']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_i']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_i']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_i']).to(torch.float32).to(device))
        self.gates_sublstm.i_gate_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_i']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_o']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_o']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_o']).to(torch.float32).to(device))
        self.gates_sublstm.o_gate_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_o']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_x.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_x_in']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_x.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_x_in']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_h.weight = \
            nn.Parameter(torch.tensor(info['w_gatessublstm_h_in']).to(torch.float32).to(device))
        self.gates_sublstm.input_w_h.bias = \
            nn.Parameter(torch.tensor(info['b_gatessublstm_h_in']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = nn.Linear(self.hidden_size, self.output_size)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x, h_t, c_t, model_params):
        w_x_t = model_params['w_x_t']
        b_x_t = model_params['b_x_t']
        w_h_t = model_params['w_h_t']
        b_h_t = model_params['b_h_t']
        
        w_added1 = model_params['w_added1']
        b_added1 = model_params['b_added1']
        
        w_added_to_out = model_params['w_added_to_out']
        b_added_to_out = model_params['b_added_to_out']
        
        w_gatessublstm_x_f = model_params['w_gatessublstm_x_f']
        b_gatessublstm_x_f = model_params['b_gatessublstm_x_f']
        w_gatessublstm_h_f = model_params['w_gatessublstm_h_f']
        b_gatessublstm_h_f = model_params['b_gatessublstm_h_f']
        w_gatessublstm_x_i = model_params['w_gatessublstm_x_i']
        b_gatessublstm_x_i = model_params['b_gatessublstm_x_i']
        w_gatessublstm_h_i = model_params['w_gatessublstm_h_i']
        b_gatessublstm_h_i = model_params['b_gatessublstm_h_i']
        w_gatessublstm_x_o = model_params['w_gatessublstm_x_o']
        b_gatessublstm_x_o = model_params['b_gatessublstm_x_o']
        w_gatessublstm_h_o = model_params['w_gatessublstm_h_o']
        b_gatessublstm_h_o = model_params['b_gatessublstm_h_o']
        w_gatessublstm_x_in = model_params['w_gatessublstm_x_in']
        b_gatessublstm_x_in = model_params['b_gatessublstm_x_in']
        w_gatessublstm_h_in = model_params['w_gatessublstm_h_in']
        b_gatessublstm_h_in = model_params['b_gatessublstm_h_in']
        
        w_l1 = model_params['w_l1']
        b_l1 = model_params['b_l1']
        
        if self.t_gate_memory == 'hidden':
            t_t_pre_act = np.dot(w_x_t, x) + b_x_t[:, None] + np.dot(w_h_t, h_t) + b_h_t[:, None]
        elif self.t_gate_memory == 'cell':
            t_t_pre_act = np.dot(w_x_t, x) + b_x_t[:, None] + np.dot(w_h_t, c_t) + b_h_t[:, None]
        t_t = 1/(1 + np.exp(-t_t_pre_act))
        
        x_t_gated = np.multiply(t_t, x)
        
        # added feedforward layer
        x_t_pre_act = np.dot(w_added1, x_t_gated) + b_added1[:, None]
        x_t = 1/(1 + np.exp(-x_t_pre_act))
        
        i_t_pre_act = np.dot(w_gatessublstm_x_i, x_t) + b_gatessublstm_x_i[:, None] + \
            np.dot(w_gatessublstm_h_i, h_t) + b_gatessublstm_h_i[:, None]
        i_t = 1/(1 + np.exp(-i_t_pre_act))
        
        f_t_pre_act = np.dot(w_gatessublstm_x_f, x_t) + b_gatessublstm_x_f[:, None] + \
            np.dot(w_gatessublstm_h_f, h_t) + b_gatessublstm_h_f[:, None]
        f_t = 1/(1 + np.exp(-f_t_pre_act))
        
        out_t_pre_act = np.dot(w_gatessublstm_x_o, x_t) + b_gatessublstm_x_o[:, None] + \
            np.dot(w_gatessublstm_h_o, h_t) + b_gatessublstm_h_o[:, None]
        out_t = 1/(1 + np.exp(-out_t_pre_act))

        g_t_pre_act = np.dot(w_gatessublstm_x_in, x_t) + b_gatessublstm_x_in[:, None] + \
            np.dot(w_gatessublstm_h_in, h_t) + b_gatessublstm_h_in[:, None]
        g_t = 1/(1 + np.exp(-g_t_pre_act))
        
        c_t = np.multiply(f_t, c_t) + g_t - i_t
        
        h_t = 1/(1 + np.exp(-c_t)) - out_t
        
        o_t_pre_act = np.dot(w_l1, h_t) + b_l1[:, None] + np.dot(w_added_to_out, x_t) + b_added_to_out[:, None]
        o_t = 1/(1 + np.exp(-o_t_pre_act))
            
        return o_t, o_t_pre_act, h_t, c_t, out_t, out_t_pre_act, g_t, g_t_pre_act, f_t, f_t_pre_act, i_t, i_t_pre_act, \
            x_t, x_t_pre_act, x_t_gated, t_t, t_t_pre_act
    
    def np_forward_loop(self, inputs):
        np_model_params = {'w_x_t': self.t_gate_w_x.weight.detach().cpu().numpy(),
                           'b_x_t': self.t_gate_w_x.bias.detach().cpu().numpy(),
                           'w_h_t': self.t_gate_w_h.weight.detach().cpu().numpy(),
                           'b_h_t': self.t_gate_w_h.bias.detach().cpu().numpy(),
                           'w_added1': self.added_l1.weight.detach().cpu().numpy(),
                           'b_added1': self.added_l1.bias.detach().cpu().numpy(),
                           'w_added_to_out': self.added_to_out.weight.detach().cpu().numpy(),
                           'b_added_to_out': self.added_to_out.bias.detach().cpu().numpy(),
                           'w_gatessublstm_x_f': self.gates_sublstm.f_gate_w_x.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_x_f': self.gates_sublstm.f_gate_w_x.bias.detach().cpu().numpy(), 
                           'w_gatessublstm_h_f': self.gates_sublstm.f_gate_w_h.weight.detach().cpu().numpy(),
                           'b_gatessublstm_h_f': self.gates_sublstm.f_gate_w_h.bias.detach().cpu().numpy(), 
                           'w_gatessublstm_x_i': self.gates_sublstm.i_gate_w_x.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_x_i': self.gates_sublstm.i_gate_w_x.bias.detach().cpu().numpy(),
                           'w_gatessublstm_h_i': self.gates_sublstm.i_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_h_i': self.gates_sublstm.i_gate_w_h.bias.detach().cpu().numpy(), 
                           'w_gatessublstm_x_o': self.gates_sublstm.o_gate_w_x.weight.detach().cpu().numpy(),
                           'b_gatessublstm_x_o': self.gates_sublstm.o_gate_w_x.bias.detach().cpu().numpy(), 
                           'w_gatessublstm_h_o': self.gates_sublstm.o_gate_w_h.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_h_o': self.gates_sublstm.o_gate_w_h.bias.detach().cpu().numpy(),
                           'w_gatessublstm_x_in': self.gates_sublstm.input_w_x.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_x_in': self.gates_sublstm.input_w_x.bias.detach().cpu().numpy(), 
                           'w_gatessublstm_h_in': self.gates_sublstm.input_w_h.weight.detach().cpu().numpy(), 
                           'b_gatessublstm_h_in': self.gates_sublstm.input_w_h.bias.detach().cpu().numpy(), 
                           'w_l1': self.l1.weight.detach().cpu().numpy(), 'b_l1': self.l1.bias.detach().cpu().numpy()}
        
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for t in range(inputs.size(2)): # loop over time bins in each cochleagram
            #print(t)
            if t % self.stim_size == 0:
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))

            input_t = inputs[:, :, t].view(self.input_size, -1)
            input_t = input_t.detach().cpu().numpy()
            
            o, o_pre_act, h, c, out, out_pre_act, g, g_pre_act, f, f_pre_act, i, i_pre_act, x_t, x_t_pre_act, x_t_gated, \
                t_gate, t_gate_pre_act = self.np_forward(input_t, h, c, np_model_params)

            if t == 0:
                t_pre_act_trace = np.array(t_gate_pre_act)
                t_trace = np.array(t_gate)
                x_t_gated_trace = np.array(x_t_gated)
                x_t_pre_act_trace = np.array(x_t_pre_act)
                x_t_trace = np.array(x_t)
                i_pre_act_trace = np.array(i_pre_act)
                i_trace = np.array(i)
                f_pre_act_trace = np.array(f_pre_act)
                f_trace = np.array(f)
                g_pre_act_trace = np.array(g_pre_act)
                g_trace = np.array(g)
                out_pre_act_trace = np.array(out_pre_act)
                out_trace = np.array(out)
                c_trace = np.array(c)
                h_trace = np.array(h)
                o_pre_act_trace = np.array(o_pre_act)
                o_trace = np.array(o)
            else:
                t_pre_act_trace = np.append(t_pre_act_trace, t_gate_pre_act, axis=1)
                t_trace = np.append(t_trace, t_gate, axis=1)
                x_t_gated_trace = np.append(x_t_gated_trace, x_t_gated, axis=1)
                x_t_pre_act_trace = np.append(x_t_pre_act_trace, x_t_pre_act, axis=1)
                x_t_trace = np.append(x_t_trace, x_t, axis=1)
                i_pre_act_trace = np.append(i_pre_act_trace, i_pre_act, axis=1)
                i_trace = np.append(i_trace, i, axis=1)
                f_pre_act_trace = np.append(f_pre_act_trace, f_pre_act, axis=1)
                f_trace = np.append(f_trace, f, axis=1)
                g_pre_act_trace = np.append(g_pre_act_trace, g_pre_act, axis=1)
                g_trace = np.append(g_trace, g, axis=1)
                out_pre_act_trace = np.append(out_pre_act_trace, out_pre_act, axis=1)
                out_trace = np.append(out_trace, out, axis=1)
                c_trace = np.append(c_trace, c, axis=1)
                h_trace = np.append(h_trace, h, axis=1)
                o_pre_act_trace = np.append(o_pre_act_trace, o_pre_act, axis=1)
                o_trace = np.append(o_trace, o, axis=1)
        
        traces = {'t_pre_act': t_pre_act_trace, 't': t_trace, 'x_t_gated': x_t_gated_trace, 'x_t_pre_act': x_t_pre_act_trace, 
                  'x_t': x_t_trace,'i_pre_act': i_pre_act_trace, 'i': i_trace, 'f_pre_act': f_pre_act_trace, 'f': f_trace, 
                  'g_pre_act': g_pre_act_trace, 'g': g_trace, 'out_pre_act': out_pre_act_trace, 'out': out_trace, 
                  'c': c_trace, 'h': h_trace, 'o_pre_act': o_pre_act_trace, 'o': o_trace}
        
        return traces
    
################################### PENNINGTON AND DAVID 2023 MODELS ########################################################
    
class pop_LN(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        '''Pennington and David params (bin_size = 10):
        self.n_F = 18
        self.n_filters_1 = 120
        self.filter_len_1 = 25
        self.n_out = 849'''
        
        self.n_F = params['n_F'] # number of frequency channels
        self.n_filters_1 = params['n_filters_1'] # number of filters in conv layer
        self.filter_len_1 = params['filter_len_1'] # length (in time bins) of filters in conv layer
        self.n_out = params['n_out'] # output (linear) layer size

        self.conv1 = torch.nn.Conv1d(self.n_F, self.n_filters_1, self.filter_len_1) # conv layer
        self.l1 = torch.nn.Linear(self.n_filters_1, self.n_out) # linear layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        self.resp_size = params['resp_size'] # response length (in time bins)

    def forward(self, x, stage):
        # forward pass of data through network
        # input x dimensions -> (f, t) or (n, f, t) where n is number of stimuli

        x = self.conv1(x) # convolutional layer, output dimensions -> (filters, t) or (n, filters, t) with n as above

        if len(x.shape) == 2: # if no stimulus dimension, permute conv layer output to (t, filters)
            x = x.permute(1, 0)
        if len(x.shape) == 3: # if there is a stimulus dimension, permute conv layer output to (n, t, filters)
            x = x.permute(0, 2, 1)
            
        # linear layer, output dimensions -> (t, n_out) or (n, t, n_out) with n as above and usually n_out = neurons
        x = self.l1(x)
        x = torch.sigmoid(x) # sigmoid nonlinearity

        return x
    
    def forward_loop(self, inputs, target_data, stage):
        # difference between stimulus and response size (may be due to zero-padding and tensorisation history having 
        # different lengths)
        stim_resp_diff = self.stim_size - self.resp_size

        # de-tensorise cochleagrams: only keep final element on history axis at each time point, then remove history 
        # dimension; tensorisation is useless for convolutional models as they take stimulus history into account through
        # filter length
        inputs = torch.squeeze(inputs[:, -1, :])

        # if input length is larger than the length of a stimulus (minibatch made up of more than one stimulus), reshape
        # input from dimensions (f, concatenated t) to (n, f, t), where n is the number of stimuli in the input minibatch
        # NOTE: models only work if minibatches include integer numbers of stimulus/response pairs
        if inputs.size(1) > self.stim_size:
            inputs = inputs.view(self.n_F, -1, self.stim_size) # reshape to (f, n, t)
            inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)

        # forward pass -> get response prediction output (dimensions -> (t, neurons) or (n, t, neurons))
        outputs = self.forward(inputs, stage)
        
        target_data = target_data.transpose(0, 1).to(torch.float32) # transpose target data to (t, neurons)

        # if target length is larger than the length of a response (minibatch made up of more than one responses), reshape
        # target from dimensions (concatenated t, neurons) to (n, t, neurons), where n is the number of responses in the 
        # target minibatch
        if target_data.size(0) > self.resp_size:
            targets = target_data.view(int(target_data.size(0)/self.resp_size), self.resp_size, -1)
            
            # if target length (in time) is larger than output length (may be due to mismatch in zero_padding bins and 
            # convolutional filter length), remove bins at the start of time axis corresponding to convolutional filter 
            # lengths while taking stimulus-response size difference into account
            if targets.size(1) > outputs.size(1):
                targets = targets[:, self.filter_len_1-1-stim_resp_diff:, :]
                
            # maybe must add option where output length is larger than target length, could be due to zero-padding being
            # being longer than convolutional filter length - would have to clip first few bins off output in that case
        
        # if target length is equal to response length but larger than output length, also adjust time axis accordingly as
        # above
        elif target_data.size(0) == self.resp_size and target_data.size(0) > outputs.size(0):
            targets = target_data[self.filter_len_1-1-stim_resp_diff:, :]
            
        # if target length is equal to output length, leave target unchanged
        elif target_data.size(0) == self.resp_size and target_data.size(0) == outputs.size(0):
            targets = target_data

        return outputs, targets
            
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.conv1.weight.norm(p=1) + self.l1.weight.norm(p=1))
        
        return regul_loss
    
    def get_params(self):
        model_info = {
            'w_conv1': self.conv1.weight, # weights - convolutional layer
            'b_conv1': self.conv1.bias, # biases - convolutional layer
            'w_l1': self.l1.weight, # weights - linear layer
            'b_l1': self.l1.bias} # biases - linear layer
        
        return model_info
    
    def reload(self, info):
        # reload convolutional layer
        self.conv1 = torch.nn.Conv1d(self.n_F, self.n_filters_1, self.filter_len_1)
        self.conv1.weight = nn.Parameter(torch.tensor(info['w_conv1']).to(torch.float32).to(device))
        self.conv1.bias = nn.Parameter(torch.tensor(info['b_conv1']).to(torch.float32).to(device))
        
        # reload linear layer
        self.l1 = torch.nn.Linear(self.n_filters_1, self.n_out)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
    def np_forward(self, x):
        w_conv1 = self.conv1.weight.detach().cpu().numpy()
        b_conv1 = self.conv1.bias.detach().cpu().numpy()
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        
        if len(x.shape) == 2:
            x = np.expand_dims(x, 0)
        
        conv_out_t = np.zeros((np.size(x, 0), self.n_filters_1, self.stim_size+1-self.filter_len_1))
        for clip in range(np.size(x, 0)):
            for filt in range(self.n_filters_1):
                for t in range(self.filter_len_1-1, self.stim_size):
                    in_x = x[clip, :, t-self.filter_len_1+1:t+1]
                    conv_prod = np.sum(np.multiply(w_conv1[filt], in_x))
                    
                    conv_out_t[clip, filt, t-self.filter_len_1+1] = conv_prod + b_conv1[filt]
        
        o_pre_act_trace = np.zeros((np.size(conv_out_t, 0), self.n_out, np.size(conv_out_t, -1)))
        o_trace = np.zeros((np.size(conv_out_t, 0), self.n_out, np.size(conv_out_t, -1)))
        
        for clip in range(np.size(x, 0)):
            x_in = conv_out_t[clip]
            o_t_pre_act = np.dot(w_l1, x_in) + b_l1[:, None]
            o_t = 1/(1 + np.exp(-o_t_pre_act))
            
            o_pre_act_trace[clip] = o_t_pre_act
            o_trace[clip] = o_t
                
        traces = {'conv_out': conv_out_t, 'o_pre_act': o_pre_act_trace, 'o': o_trace}
                
        return traces
    
    def np_forward_loop(self, inputs):
        inputs = torch.squeeze(inputs[:, -1, :])
        
        if inputs.size(1) > self.stim_size:
            inputs = inputs.view(self.n_F, -1, self.stim_size) # reshape to (f, n, t)
            inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)
        
        inputs = inputs.detach().cpu().numpy()
        
        traces = self.np_forward(inputs)

        return traces

#############################################################################################################################
    
class oneD_CNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        '''Pennington and David params (bin_size = 10):
        self.n_F = 18
        self.n_filters_1 = 100
        self.filter_len_1 = 25
        self.n_HU_1 = 120
        self.n_out = 849'''
        
        self.n_F = params['n_F'] # number of frequency channels
        self.n_filters_1 = params['n_filters_1'] # number of filters in conv layer
        self.filter_len_1 = params['filter_len_1'] # length (in time bins) of filters in conv layer
        self.n_HU_1 = params['n_HU_1'] # number of hidden units in 2nd linear layer
        self.n_out = params['n_out'] # output (linear) layer size
        
        self.conv1 = torch.nn.Conv1d(self.n_F, self.n_filters_1, self.filter_len_1) # conv layer
        self.l1 = torch.nn.Linear(self.n_filters_1, self.n_HU_1) # 1st linear layer
        self.l2 = torch.nn.Linear(self.n_HU_1, self.n_out) # 2nd linear layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        self.resp_size = params['resp_size'] # response length (in time bins)
        
        if 'split' in params:
            self.split = params['split']
        else:
            self.split = 1

    def forward(self, x, stage):
        # forward pass of data through network
        # input x dimensions -> (f, t) or (n, f, t) where n is number of stimuli

        x = self.conv1(x) # convolutional layer, output dimensions -> (filters, t) or (n, filters, t) with n as above
        x = torch.sigmoid(x) # sigmoid nonlinearity
        
        if len(x.shape) == 2: # if no stimulus dimension, permute conv layer output to (t, filters)
            x = x.permute(1, 0)
        elif len(x.shape) == 3: # if there is a stimulus dimension, permute conv layer output to (n, t, filters)
            x = x.permute(0, 2, 1)

        # linear layer, output dimensions -> (t, n_HU_1) or (n, t, n_HU_1) with n as above
        x = self.l1(x)
        x = torch.sigmoid(x)

        # linear layer, output dimensions -> (t, n_out) or (n, t, n_out) with n as above and usually n_out = neurons
        x = self.l2(x)
        x = torch.sigmoid(x)

        return x
    
    def forward_loop(self, inputs, target_data, stage):
        if stage == 'val' or stage == 'test':
            n_data = int(np.floor(target_data.size(1)/(self.resp_size/self.split)))
            stim_size = int(inputs.size(2)/n_data)
            resp_size = int(target_data.size(1)/n_data)
        elif stage =='train':
            stim_size = self.stim_size
            resp_size = self.resp_size
        
        # difference between stimulus and response size (may be due to zero-padding and tensorisation history having 
        # different lengths)
        stim_resp_diff = stim_size - resp_size
        
        # de-tensorise cochleagrams: only keep final element on history axis at each time point, then remove history 
        # dimension; tensorisation is useless for convolutional models as they take stimulus history into account through
        # filter length
        inputs = torch.squeeze(inputs[:, -1, :])

        # if input length is larger than the length of a stimulus (minibatch made up of more than one stimulus), reshape
        # input from dimensions (f, concatenated t) to (n, f, t), where n is the number of stimuli in the input minibatch
        # NOTE: models only work if minibatches include integer numbers of stimulus/response pairs
        if inputs.size(1) > stim_size:
            if inputs.size(1) % stim_size == 0: # if batch includes an integer number of stimulus/response pairs
                inputs = inputs.view(self.n_F, -1, stim_size) # reshape to (f, n, t)
                inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)
            else: # this option is only used in the split data analysis
                diff = inputs.size(1) % stim_size
                inputs = inputs[:, :-diff]
                inputs = inputs.view(self.n_F, -1, stim_size) # reshape to (f, n, t)
                inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)
        
        # forward pass -> get response prediction output (dimensions -> (t, neurons) or (n, t, neurons))
        outputs = self.forward(inputs, stage)

        target_data = target_data.transpose(0, 1).to(torch.float32) # transpose target data to (t, neurons)

        # if target length is larger than the length of a response (minibatch made up of more than one responses), reshape
        # target from dimensions (concatenated t, neurons) to (n, t, neurons), where n is the number of responses in the 
        # target minibatch
        if target_data.size(0) > resp_size:
            if target_data.size(0) % resp_size == 0: # if batch includes an integer number of stimulus/response pairs
                targets = target_data.view(int(target_data.size(0)/resp_size), resp_size, -1)
            else: # this option is only used in the split data analysis
                diff = target_data.size(0) % resp_size
                target_data = target_data[:-diff]
                targets = target_data.view(int(target_data.size(0)/resp_size), resp_size, -1)
            
            # if target length (in time) is larger than output length (may be due to mismatch in zero_padding bins and 
            # convolutional filter length), remove bins at the start of time axis corresponding to convolutional filter 
            # lengths while taking stimulus-response size difference into account
            if targets.size(1) > outputs.size(1):
                #targets = targets[:, self.filter_len_1-1-stim_resp_diff:, :]
                
                targets_new = torch.zeros_like(targets[:, self.filter_len_1-1-stim_resp_diff:, :])
                if stage == 'val' or stage == 'test':
                    n_resps = int(targets.size(0)*self.split)
                    splits_in_resp = int(targets.size(0)/n_resps)
                    
                    for sID in range(n_resps):
                        for rID in range(splits_in_resp):
                            start_lag = rID*stim_resp_diff
                            end_lag = int((targets.size(0)/n_resps - rID - 1)*stim_resp_diff)
                            
                            if end_lag == 0:
                                targets_new[sID*splits_in_resp+rID] = targets[sID*splits_in_resp+rID, start_lag:]
                            else:
                                targets_new[sID*splits_in_resp+rID] = targets[sID*splits_in_resp+rID, start_lag:-end_lag]
                                
                elif stage == 'train':
                    for rID in range(targets.size(0)):
                        start_lag = rID*stim_resp_diff
                        end_lag = (targets.size(0) - rID - 1)*stim_resp_diff

                        if end_lag == 0:
                            targets_new[rID] = targets[rID, start_lag:]
                        else:
                            targets_new[rID] = targets[rID, start_lag:-end_lag]
                            
                targets = targets_new
            # maybe must add option where output length is larger than target length, could be due to zero-padding being
            # being longer than convolutional filter length - would have to clip first few bins off output in that case
        
        # if target length is equal to response length but larger than output length, also adjust time axis accordingly as
        # above
        elif target_data.size(0) == resp_size and target_data.size(0) > outputs.size(0):
            targets = target_data[self.filter_len_1-1-stim_resp_diff:, :]
            
        # if target length is equal to output length, leave target unchanged
        elif target_data.size(0) == resp_size and target_data.size(0) == outputs.size(0):
            targets = target_data

        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.conv1.weight.norm(p=1) + self.l1.weight.norm(p=1) + self.l2.weight.norm(p=1))
        
        return regul_loss

    def get_params(self):
        model_info = {
            'w_conv1': self.conv1.weight, # weights - convolutional layer
            'b_conv1': self.conv1.bias, # biases - convolutional layer
            'w_l1': self.l1.weight, # weights - 1st linear layer
            'b_l1': self.l1.bias, # biases - 1st linear layer
            'w_l2': self.l2.weight, # weights - 2nd linear layer
            'b_l2': self.l2.bias} # biases - 2nd linear layer
        
        return model_info
    
    def reload(self, info):
        # reload convolutional layer
        self.conv1 = torch.nn.Conv1d(self.n_F, self.n_filters_1, self.filter_len_1)
        self.conv1.weight = nn.Parameter(torch.tensor(info['w_conv1']).to(torch.float32).to(device))
        self.conv1.bias = nn.Parameter(torch.tensor(info['b_conv1']).to(torch.float32).to(device))
        
        # reload first linear layer
        self.l1 = torch.nn.Linear(self.n_filters_1, self.n_HU_1)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
        # reload second linear layer
        self.l2 = torch.nn.Linear(self.n_HU_1, self.n_out)
        self.l2.weight = nn.Parameter(torch.tensor(info['w_l2']).to(torch.float32).to(device))
        self.l2.bias = nn.Parameter(torch.tensor(info['b_l2']).to(torch.float32).to(device))
        
    def np_forward(self, x):
        w_conv1 = self.conv1.weight.detach().cpu().numpy()
        b_conv1 = self.conv1.bias.detach().cpu().numpy()
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        w_l2 = self.l2.weight.detach().cpu().numpy()
        b_l2 = self.l2.bias.detach().cpu().numpy()
        
        if len(x.shape) == 2:
            x = np.expand_dims(x, 0)
        
        conv_out_t_pre_act = np.zeros((np.size(x, 0), self.n_filters_1, self.stim_size+1-self.filter_len_1))
        for clip in range(np.size(x, 0)):
            for filt in range(self.n_filters_1):
                for t in range(self.filter_len_1-1, self.stim_size):
                    in_x = x[clip, :, t-self.filter_len_1+1:t+1]
                    conv_prod = np.sum(np.multiply(w_conv1[filt], in_x))
                    
                    conv_out_t_pre_act[clip, filt, t-self.filter_len_1+1] = conv_prod + b_conv1[filt]
        
        conv_out_t = 1/(1 + np.exp(-conv_out_t_pre_act))
        
        h_pre_act_trace = np.zeros((np.size(conv_out_t, 0), self.n_HU_1, np.size(conv_out_t, -1)))
        h_trace = np.zeros((np.size(conv_out_t, 0), self.n_HU_1, np.size(conv_out_t, -1)))
        
        o_pre_act_trace = np.zeros((np.size(conv_out_t, 0), self.n_out, np.size(conv_out_t, -1)))
        o_trace = np.zeros((np.size(conv_out_t, 0), self.n_out, np.size(conv_out_t, -1)))
        
        for clip in range(np.size(x, 0)):
            x_in = conv_out_t[clip]
            
            h_t_pre_act = np.dot(w_l1, x_in) + b_l1[:, None]
            h_t = 1/(1 + np.exp(-h_t_pre_act))
            
            o_t_pre_act = np.dot(w_l2, h_t) + b_l2[:, None]
            o_t = 1/(1 + np.exp(-o_t_pre_act))
            
            h_pre_act_trace[clip] = h_t_pre_act
            h_trace[clip] = h_t
            o_pre_act_trace[clip] = o_t_pre_act
            o_trace[clip] = o_t
                
        traces = {'conv_out_pre_act': conv_out_t_pre_act, 'conv_out': conv_out_t, 'h_pre_act': h_pre_act_trace, 'h': h_trace, 
                  'o_pre_act': o_pre_act_trace, 'o': o_trace}
                
        return traces
    
    def np_forward_loop(self, inputs):
        inputs = torch.squeeze(inputs[:, -1, :])
        
        if inputs.size(1) > self.stim_size:
            inputs = inputs.view(self.n_F, -1, self.stim_size) # reshape to (f, n, t)
            inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)
        
        inputs = inputs.detach().cpu().numpy()
        
        traces = self.np_forward(inputs)

        return traces

#############################################################################################################################

class oneD_x2_CNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        '''Pennington and David params (bin_size = 10):
        self.n_F = 18
        self.n_filters_1 = 70
        self.n_filters_2 = 80
        self.filter_len_1 = 15
        self.filter_len_2 = 10
        self.n_HU_1 = 100
        self.n_out = 849'''
        
        self.n_F = params['n_F'] # number of frequency channels
        self.n_filters_1 = params['n_filters_1'] # number of filters in 1st conv layer
        self.n_filters_2 = params['n_filters_2'] # number of filters in 2nd conv layer
        self.filter_len_1 = params['filter_len_1'] # length (in time bins) of filters in 1st conv layer
        self.filter_len_2 = params['filter_len_2'] # length (in time bins) of filters in 2nd conv layer
        self.n_HU_1 = params['n_HU_1'] # number of hidden units in 1st linear layer
        self.n_out = params['n_out'] # output (linear) layer size

        self.conv1 = torch.nn.Conv1d(self.n_F, self.n_filters_1, self.filter_len_1) # 1st conv layer
        self.conv2 = torch.nn.Conv1d(self.n_filters_1, self.n_filters_2, self.filter_len_2) # 2nd conv layer
        self.l1 = torch.nn.Linear(self.n_filters_2, self.n_HU_1) # 1st linear layer
        self.l2 = torch.nn.Linear(self.n_HU_1, self.n_out) # 2nd linear layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        self.resp_size = params['resp_size'] # response length (in time bins)

    def forward(self, x, stage):
        # forward pass of data through network
        # input x dimensions -> (f, t) or (n, f, t) where n is number of stimuli

        x = self.conv1(x) # convolutional layer, output dimensions -> (filters, t) or (n, filters, t) with n as above
        x = torch.sigmoid(x) # sigmoid nonlinearity

        x = self.conv2(x) # convolutional layer, output dimensions -> (filters, t) or (n, filters, t) with n as above
        x = torch.sigmoid(x) # sigmoid nonlinearity

        if len(x.shape) == 2: # if no stimulus dimension, permute conv layer output to (t, filters)
            x = x.permute(1, 0)
        elif len(x.shape) == 3: # if there is a stimulus dimension, permute conv layer output to (n, t, filters)
            x = x.permute(0, 2, 1)

        # linear layer, output dimensions -> (t, n_HU_1) or (n, t, n_HU_1) with n as above
        x = self.l1(x)
        x = torch.sigmoid(x) # sigmoid nonlinearity

        # linear layer, output dimensions -> (t, n_out) or (n, t, n_out) with n as above and usually n_out = neurons
        x = self.l2(x)
        x = torch.sigmoid(x) # sigmoid nonlinearity

        return x
    
    def forward_loop(self, inputs, target_data, stage):
        # difference between stimulus and response size (may be due to zero-padding and tensorisation history having 
        # different lengths)
        stim_resp_diff = self.stim_size - self.resp_size

        # de-tensorise cochleagrams: only keep final element on history axis at each time point, then remove history 
        # dimension; tensorisation is useless for convolutional models as they take stimulus history into account through
        # filter length
        inputs = torch.squeeze(inputs[:, -1, :])

        # if input length is larger than the length of a stimulus (minibatch made up of more than one stimulus), reshape
        # input from dimensions (f, concatenated t) to (n, f, t), where n is the number of stimuli in the input minibatch
        # NOTE: models only work if minibatches include integer numbers of stimulus/response pairs
        if inputs.size(1) > self.stim_size:
            inputs = inputs.view(self.n_F, -1, self.stim_size) # reshape to (f, n, t)
            inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)
        
        # forward pass -> get response prediction output (dimensions -> (t, neurons) or (n, t, neurons))
        outputs = self.forward(inputs, stage)

        target_data = target_data.transpose(0, 1).to(torch.float32) # transpose target data to (t, neurons)
        
        # if target length is larger than the length of a response (minibatch made up of more than one responses), reshape
        # target from dimensions (concatenated t, neurons) to (n, t, neurons), where n is the number of responses in the 
        # target minibatch
        if target_data.size(0) > self.resp_size:
            targets = target_data.view(int(target_data.size(0)/self.resp_size), self.resp_size, -1)
            
            # if target length (in time) is larger than output length (may be due to mismatch in zero_padding bins and 
            # convolutional filter length), remove bins at the start of time axis corresponding to convolutional filter 
            # lengths while taking stimulus-response size difference into account
            if targets.size(1) > outputs.size(1):
                targets = targets[:, self.filter_len_1+self.filter_len_2-2-stim_resp_diff:, :]
            
            # maybe must add option where output length is larger than target length, could be due to zero-padding being
            # being longer than convolutional filter length - would have to clip first few bins off output in that case
        
        # if target length is equal to response length but larger than output length, also adjust time axis accordingly as
        # above
        elif target_data.size(0) == self.resp_size and target_data.size(0) > outputs.size(0):
            targets = target_data[self.filter_len_1+self.filter_len_2-2-stim_resp_diff:, :]
        
        # if target length is equal to output length, leave target unchanged
        elif target_data.size(0) == self.resp_size and target_data.size(0) == outputs.size(0):
            targets = target_data

        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.conv1.weight.norm(p=1) + self.conv2.weight.norm(p=1) + \
                           self.l1.weight.norm(p=1) + self.l2.weight.norm(p=1))
        
        return regul_loss
        
    def get_params(self):
        model_info = {
            'w_conv1': self.conv1.weight, # weights - 1st convolutional layer
            'b_conv1': self.conv1.bias, # biases - 1st convolutional layer
            'w_conv2': self.conv2.weight, # weights - 2nd convolutional layer
            'b_conv2': self.conv2.bias, # biases - 2nd convolutional layer
            'w_l1': self.l1.weight, # weights - 1st linear layer
            'b_l1': self.l1.bias, # biases - 1st linear layer
            'w_l2': self.l2.weight, # weights - 2nd linear layer
            'b_l2': self.l2.bias} # biases - 2nd linear layer
        
        return model_info
    
    def reload(self, info):
        # reload first convolutional layer
        self.conv1 = torch.nn.Conv1d(self.n_F, self.n_filters_1, self.filter_len_1)
        self.conv1.weight = nn.Parameter(torch.tensor(info['w_conv1']).to(torch.float32).to(device))
        self.conv1.bias = nn.Parameter(torch.tensor(info['b_conv1']).to(torch.float32).to(device))
        
        # reload second convolutional layer
        self.conv2 = torch.nn.Conv1d(self.n_filters_1, self.n_filters_2, self.filter_len_2)
        self.conv2.weight = nn.Parameter(torch.tensor(info['w_conv2']).to(torch.float32).to(device))
        self.conv2.bias = nn.Parameter(torch.tensor(info['b_conv2']).to(torch.float32).to(device))
        
        # reload first linear layer
        self.l1 = torch.nn.Linear(self.n_filters_2, self.n_HU_1)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
        # reload second linear layer
        self.l2 = torch.nn.Linear(self.n_HU_1, self.n_out)
        self.l2.weight = nn.Parameter(torch.tensor(info['w_l2']).to(torch.float32).to(device))
        self.l2.bias = nn.Parameter(torch.tensor(info['b_l2']).to(torch.float32).to(device))
        
    def np_forward(self, x):
        w_conv1 = self.conv1.weight.detach().cpu().numpy()
        b_conv1 = self.conv1.bias.detach().cpu().numpy()
        w_conv2 = self.conv2.weight.detach().cpu().numpy()
        b_conv2 = self.conv2.bias.detach().cpu().numpy()
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        w_l2 = self.l2.weight.detach().cpu().numpy()
        b_l2 = self.l2.bias.detach().cpu().numpy()
        
        if len(x.shape) == 2:
            x = np.expand_dims(x, 0)
        
        conv1_out_t_pre_act = np.zeros((np.size(x, 0), self.n_filters_1, self.stim_size+1-self.filter_len_1))
        for clip in range(np.size(x, 0)):
            for filt in range(self.n_filters_1):
                for t in range(self.filter_len_1-1, self.stim_size):
                    in_x = x[clip, :, t-self.filter_len_1+1:t+1]
                    conv_prod = np.sum(np.multiply(w_conv1[filt], in_x))
                    
                    conv1_out_t_pre_act[clip, filt, t-self.filter_len_1+1] = conv_prod + b_conv1[filt]
        
        conv1_out_t = 1/(1 + np.exp(-conv1_out_t_pre_act))
        
        conv2_out_t_pre_act = np.zeros((np.size(x, 0), self.n_filters_2, np.size(conv1_out_t, -1)+1-self.filter_len_2))
        for clip in range(np.size(x, 0)):
            for filt in range(self.n_filters_2):
                for t in range(self.filter_len_2-1, np.size(conv1_out_t, -1)):
                    in_x = conv1_out_t[clip, :, t-self.filter_len_2+1:t+1]
                    conv_prod = np.sum(np.multiply(w_conv2[filt], in_x))
                    
                    conv2_out_t_pre_act[clip, filt, t-self.filter_len_2+1] = conv_prod + b_conv2[filt]
        
        conv2_out_t = 1/(1 + np.exp(-conv2_out_t_pre_act))            
        
        h_pre_act_trace = np.zeros((np.size(conv2_out_t, 0), self.n_HU_1, np.size(conv2_out_t, -1)))
        h_trace = np.zeros((np.size(conv2_out_t, 0), self.n_HU_1, np.size(conv2_out_t, -1)))
        
        o_pre_act_trace = np.zeros((np.size(conv2_out_t, 0), self.n_out, np.size(conv2_out_t, -1)))
        o_trace = np.zeros((np.size(conv2_out_t, 0), self.n_out, np.size(conv2_out_t, -1)))
        
        for clip in range(np.size(x, 0)):
            x_in = conv2_out_t[clip]
            
            h_t_pre_act = np.dot(w_l1, x_in) + b_l1[:, None]
            h_t = 1/(1 + np.exp(-h_t_pre_act))
            
            o_t_pre_act = np.dot(w_l2, h_t) + b_l2[:, None]
            o_t = 1/(1 + np.exp(-o_t_pre_act))
            
            h_pre_act_trace[clip] = h_t_pre_act
            h_trace[clip] = h_t
            o_pre_act_trace[clip] = o_t_pre_act
            o_trace[clip] = o_t
                
        traces = {'conv1_out_pre_act': conv1_out_t_pre_act, 'conv1_out': conv1_out_t, 
                  'conv2_out_pre_act': conv2_out_t_pre_act, 'conv2_out': conv2_out_t, 'h_pre_act': h_pre_act_trace, 
                  'h': h_trace, 'o_pre_act': o_pre_act_trace, 'o': o_trace}
                
        return traces
    
    def np_forward_loop(self, inputs):
        inputs = torch.squeeze(inputs[:, -1, :])
        
        if inputs.size(1) > self.stim_size:
            inputs = inputs.view(self.n_F, -1, self.stim_size) # reshape to (f, n, t)
            inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)
        
        inputs = inputs.detach().cpu().numpy()
        
        traces = self.np_forward(inputs)

        return traces

#############################################################################################################################
    
class twoD_CNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        '''Pennington and David params (bin_size = 10):
        self.n_F = 18
        self.n_filters_1 = 10
        self.n_filters_2 = 10
        self.n_filters_3 = 10
        self.filter_area = (3, 8)
        self.n_HU_1 = 90
        self.n_out = 849'''
        
        self.n_F = params['n_F'] # number of frequency channels
        self.n_filters_1 = params['n_filters_1'] # number of filters in 1st conv layer
        self.n_filters_2 = params['n_filters_2'] # number of filters in 2nd conv layer
        self.n_filters_3 = params['n_filters_3'] # number of filters in 3rd conv layer
        self.filter_area = params['filter_area'] # area (frequency channels, time bins) of 2D conv filters
        self.n_HU_1 = params['n_HU_1'] # number of hidden units in 1st linear layer
        self.n_out = params['n_out'] # output (linear) layer size

        self.conv1 = torch.nn.Conv2d(1, self.n_filters_1, self.filter_area) # 1st conv layer
        self.conv2 = torch.nn.Conv2d(self.n_filters_1, self.n_filters_2, self.filter_area) # 2nd conv layer
        self.conv3 = torch.nn.Conv2d(self.n_filters_2, self.n_filters_3, self.filter_area) # 3rd conv layer
        self.l1 = torch.nn.Linear(self.n_filters_3*(self.n_F-3*(self.filter_area[0]-1)), self.n_HU_1) # 1st linear layer
        self.l2 = torch.nn.Linear(self.n_HU_1, self.n_out) # 2nd linear layer
        
        self.stim_size = params['stim_size'] # stimulus length (in time bins)
        self.resp_size = params['resp_size'] # response length (in time bins)
        
        if 'split' in params:
            self.split = params['split']
        else:
            self.split = 1

    def forward(self, x, stage):
        # forward pass of data through network
        # input x dimensions -> (channels, f, t) or (n, channels, f, t) where n is number of stimuli

        # convolutional layer, output dimensions -> (filters, f, t) or (n, filters, f, t) with n as above; number of
        # frequencies is reduced
        x = self.conv1(x)
        x = torch.sigmoid(x) # sigmoid nonlinearity

        # convolutional layer, output dimensions -> (filters, f, t) or (n, filters, f, t) with n as above; number of
        # frequencies is reduced
        x = self.conv2(x)
        x = torch.sigmoid(x) # sigmoid nonlinearity

        # convolutional layer, output dimensions -> (filters, f, t) or (n, filters, f, t) with n as above; number of
        # frequencies is reduced
        x = self.conv3(x)
        x = torch.sigmoid(x) # sigmoid nonlinearity

        if len(x.shape) == 3:
            x = x.permute(2, 1, 0) # permute dimensions to (t, f, filters)
            x = x.reshape(-1, x.size(1)*x.size(2)) # bring together f and filters axis
            
        if len(x.shape) == 4:
            x = x.permute(0, 3, 2, 1) # permute dimensions to (n, t, f, filters)
            x = x.reshape(x.size(0), -1, x.size(2)*x.size(3)) # bring together f and filters axis
            
        x = self.l1(x) # linear layer, output dimensions -> (t, n_HU_1) or (n, t, n_HU_1) with n as above
        x = torch.sigmoid(x) # sigmoid nonlinearity

        x = self.l2(x) # linear layer, output dimensions -> (t, n_out) or (n, t, n_out) with n as above
        x = torch.sigmoid(x) # sigmoid nonlinearity

        return x
    
    '''def forward_loop(self, inputs, target_data, stage):
        # difference between stimulus and response size (may be due to zero-padding and tensorisation history having 
        # different lengths)
        
        stim_resp_diff = self.stim_size - self.resp_size

        # de-tensorise cochleagrams: only keep final element on history axis at each time point, then remove history 
        # dimension; tensorisation is useless for convolutional models as they take stimulus history into account through
        # filter length
        inputs = torch.squeeze(inputs[:, -1, :])

        # if input length is larger than the length of a stimulus (minibatch made up of more than one stimulus), reshape
        # input from dimensions (f, concatenated t) to (n, f, t), where n is the number of stimuli in the input minibatch
        # NOTE: models only work if minibatches include integer numbers of stimulus/response pairs
        if inputs.size(1) > self.stim_size:
            if inputs.size(1) % self.stim_size == 0: # if batch includes an integer number of stimulus/response pairs
                inputs = inputs.view(self.n_F, -1, self.stim_size) # reshape to (f, n, t)
                inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)
            else: # this option is only used in the split data analysis
                diff = inputs.size(1) % self.stim_size
                inputs = inputs[:, :-diff]
                inputs = inputs.view(self.n_F, -1, self.stim_size) # reshape to (f, n, t)
                inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)

        # Conv2D filters usually work on images with a height and width, and a certain number of channels; "channels" in
        # this context does not refer to the number of frequency channels, which are instead considered as the height of a
        # cochleagram "image"; input to a conv2D layer must have dimensions (n, channels, height, width) or (channels, 
        # height, width), hence the unsqueezing below
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(0) # add dimension (channels) at axis 0 -> (channels, f, t)
        elif len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1) # add dimension (channels) at axis 1 -> (n, channels, f, t)

        outputs = self.forward(inputs, stage) # forward pass -> get response prediction output (dimensions -> (t, neurons))

        target_data = target_data.transpose(0, 1).to(torch.float32) # transpose target data to (t, neurons)

        # if target length is larger than the length of a response (minibatch made up of more than one responses), reshape
        # target from dimensions (concatenated t, neurons) to (n, t, neurons), where n is the number of responses in the 
        # target minibatch
        if target_data.size(0) > self.resp_size:
            if target_data.size(0) % self.resp_size == 0: # if batch includes an integer number of stimulus/response pairs
                targets = target_data.view(int(target_data.size(0)/self.resp_size), self.resp_size, -1)
            else: # this option is only used in the split data analysis
                diff = target_data.size(0) % self.resp_size
                target_data = target_data[:-diff]
                targets = target_data.view(int(target_data.size(0)/self.resp_size), self.resp_size, -1)
                
            # if target length (in time) is larger than output length (may be due to mismatch in zero_padding bins and 
            # convolutional filter length), remove bins at the start of time axis corresponding to convolutional filter 
            # lengths while taking stimulus-response size difference into account
            if targets.size(1) > outputs.size(1):
                #targets = targets[:, self.filter_len_1-1-stim_resp_diff:, :]
                
                targets_new = torch.zeros_like(targets[:, 3*(self.filter_area[1]-1)-stim_resp_diff:, :])
                if stage == 'val' or stage == 'test':
                    n_resps = int(targets.size(0)*self.split)
                    splits_in_resp = int(targets.size(0)/n_resps)
                    
                    for sID in range(n_resps):
                        for rID in range(splits_in_resp):
                            start_lag = rID*stim_resp_diff
                            end_lag = int((targets.size(0)/n_resps - rID - 1)*stim_resp_diff)
                            
                            if end_lag == 0:
                                targets_new[sID*splits_in_resp+rID] = targets[sID*splits_in_resp+rID, start_lag:]
                            else:
                                targets_new[sID*splits_in_resp+rID] = targets[sID*splits_in_resp+rID, start_lag:-end_lag]
                                
                elif stage == 'train':
                    for rID in range(targets.size(0)):
                        start_lag = rID*stim_resp_diff
                        end_lag = (targets.size(0) - rID - 1)*stim_resp_diff

                        if end_lag == 0:
                            targets_new[rID] = targets[rID, start_lag:]
                        else:
                            targets_new[rID] = targets[rID, start_lag:-end_lag]

                targets = targets_new
            # maybe must add option where output length is larger than target length, could be due to zero-padding being
            # being longer than convolutional filter length - would have to clip first few bins off output in that case
        
        # if target length is equal to response length but larger than output length, also adjust time axis accordingly as
        # above
        elif target_data.size(0) == self.resp_size and target_data.size(0) > outputs.size(0):
            targets = target_data[3*(self.filter_area[1]-1)-stim_resp_diff:, :]
        
        # if target length is equal to output length, leave target unchanged
        elif target_data.size(0) == self.resp_size and target_data.size(0) == outputs.size(0):
            targets = target_data

        return outputs, targets'''
    
    def forward_loop(self, inputs, target_data, stage):
        # difference between stimulus and response size (may be due to zero-padding and tensorisation history having 
        # different lengths)
        if stage == 'val' or stage == 'test':
            n_data = int(np.floor(target_data.size(1)/(self.resp_size/self.split)))
            stim_size = int(inputs.size(2)/n_data)
            resp_size = int(target_data.size(1)/n_data)

        elif stage =='train':
            stim_size = self.stim_size
            resp_size = self.resp_size
        
        stim_resp_diff = stim_size - resp_size
        
        # de-tensorise cochleagrams: only keep final element on history axis at each time point, then remove history 
        # dimension; tensorisation is useless for convolutional models as they take stimulus history into account through
        # filter length
        inputs = torch.squeeze(inputs[:, -1, :])
        
        if self.split == 0.125 and stage == 'train':
            inputs, target_data = \
                self.split_sequences(inputs, target_data, stim_size, self.filter_area[1], int(1/self.split))
        
        else:
            # if input length is larger than the length of a stimulus (minibatch made up of more than one stimulus), reshape
            # input from dimensions (f, concatenated t) to (n, f, t), where n is the number of stimuli in the input minibatch
            # NOTE: models only work if minibatches include integer numbers of stimulus/response pairs
            if inputs.size(1) > stim_size:
                if inputs.size(1) % stim_size == 0: # if batch includes an integer number of stimulus/response pairs
                    inputs = inputs.view(self.n_F, -1, stim_size) # reshape to (f, n, t)
                    inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)
                else: # this option is only used in the split data analysis
                    diff = inputs.size(1) % stim_size
                    inputs = inputs[:, :-diff]
                    inputs = inputs.view(self.n_F, -1, stim_size) # reshape to (f, n, t)
                    inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)

        # Conv2D filters usually work on images with a height and width, and a certain number of channels; "channels" in
        # this context does not refer to the number of frequency channels, which are instead considered as the height of a
        # cochleagram "image"; input to a conv2D layer must have dimensions (n, channels, height, width) or (channels, 
        # height, width), hence the unsqueezing below
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(0) # add dimension (channels) at axis 0 -> (channels, f, t)
        elif len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1) # add dimension (channels) at axis 1 -> (n, channels, f, t)

        outputs = self.forward(inputs, stage) # forward pass -> get response prediction output (dimensions -> (t, neurons))
        
        if self.split == 0.125 and stage == 'train':
            target_data = target_data.permute(1, 2, 0).to(torch.float32)
            targets = target_data
        else:
            target_data = target_data.transpose(0, 1).to(torch.float32) # transpose target data to (t, neurons)

            # if target length is larger than the length of a response (minibatch made up of more than one responses),
            # reshape target from dimensions (concatenated t, neurons) to (n, t, neurons), where n is the number of responses
            # in the target minibatch
            if target_data.size(0) > resp_size:
                if target_data.size(0) % resp_size == 0: # if batch includes an integer number of stimulus/response pairs
                    targets = target_data.view(int(target_data.size(0)/resp_size), resp_size, -1)
                else: # this option is only used in the split data analysis
                    diff = target_data.size(0) % resp_size
                    target_data = target_data[:-diff]
                    targets = target_data.view(int(target_data.size(0)/resp_size), resp_size, -1)
                    
                # if target length (in time) is larger than output length (may be due to mismatch in zero_padding bins and 
                # convolutional filter length), remove bins at the start of time axis corresponding to convolutional filter 
                # lengths while taking stimulus-response size difference into account
                if targets.size(1) > outputs.size(1):                
                    targets_new = torch.zeros_like(targets[:, 3*(self.filter_area[1]-1)-stim_resp_diff:, :])
                    if stage == 'val' or stage == 'test':
                        n_resps = int(targets.size(0)*self.split)
                        splits_in_resp = int(targets.size(0)/n_resps)
                        
                        for sID in range(n_resps):
                            for rID in range(splits_in_resp):
                                start_lag = rID*stim_resp_diff
                                end_lag = int((targets.size(0)/n_resps - rID - 1)*stim_resp_diff)
                                
                                if end_lag == 0:
                                    targets_new[sID*splits_in_resp+rID] = targets[sID*splits_in_resp+rID, start_lag:]
                                else:
                                    targets_new[sID*splits_in_resp+rID] = targets[sID*splits_in_resp+rID, start_lag:-end_lag]
                                    
                    elif stage == 'train':
                        for rID in range(targets.size(0)):
                            start_lag = rID*stim_resp_diff
                            end_lag = (targets.size(0) - rID - 1)*stim_resp_diff
    
                            if end_lag == 0:
                                targets_new[rID] = targets[rID, start_lag:]
                            else:
                                targets_new[rID] = targets[rID, start_lag:-end_lag]
    
                    targets = targets_new
                # maybe must add option where output length is larger than target length, could be due to zero-padding being
                # being longer than convolutional filter length - would have to clip first few bins off output in that case
            
            # if target length is equal to response length but larger than output length, also adjust time axis accordingly
            # as above
            elif target_data.size(0) == resp_size and target_data.size(0) > outputs.size(0):
                targets = target_data[3*(self.filter_area[1]-1)-stim_resp_diff:, :]
            
            # if target length is equal to output length, leave target unchanged
            elif target_data.size(0) == resp_size and target_data.size(0) == outputs.size(0):
                targets = target_data

        return outputs, targets
    
    def regul(self, lamb):
        # L1 regularisation loss on network weights
        regul_loss = lamb*(self.conv1.weight.norm(p=1) + self.conv2.weight.norm(p=1) + self.conv3.weight.norm(p=1) + \
                           self.l1.weight.norm(p=1) + self.l2.weight.norm(p=1))
        
        return regul_loss

    def get_params(self):
        model_info = {
            'w_conv1': self.conv1.weight, # weights - 1st convolutional layer
            'b_conv1': self.conv1.bias, # biases - 1st convolutional layer
            'w_conv2': self.conv2.weight, # weights - 2nd convolutional layer
            'b_conv2': self.conv2.bias, # biases - 2nd convolutional layer
            'w_conv3': self.conv3.weight, # weights - 3rd convolutional layer
            'b_conv3': self.conv3.bias, # biases - 3rd convolutional layer
            'w_l1': self.l1.weight, # weights - 1st linear layer
            'b_l1': self.l1.bias, # biases - 1st linear layer
            'w_l2': self.l2.weight, # weights - 2nd linear layer
            'b_l2': self.l2.bias} # biases - 2nd linear layer
        
        return model_info
    
    def reload(self, info):
        # reload first convolutional layer
        self.conv1 = torch.nn.Conv2d(1, self.n_filters_1, self.filter_area)
        self.conv1.weight = nn.Parameter(torch.tensor(info['w_conv1']).to(torch.float32).to(device))
        self.conv1.bias = nn.Parameter(torch.tensor(info['b_conv1']).to(torch.float32).to(device))
        
        # reload second convolutional layer
        self.conv2 = torch.nn.Conv2d(self.n_filters_1, self.n_filters_2, self.filter_area)
        self.conv2.weight = nn.Parameter(torch.tensor(info['w_conv2']).to(torch.float32).to(device))
        self.conv2.bias = nn.Parameter(torch.tensor(info['b_conv2']).to(torch.float32).to(device))
        
        # reload third convolutional layer
        self.conv3 = torch.nn.Conv2d(self.n_filters_2, self.n_filters_3, self.filter_area)
        self.conv3.weight = nn.Parameter(torch.tensor(info['w_conv3']).to(torch.float32).to(device))
        self.conv3.bias = nn.Parameter(torch.tensor(info['b_conv3']).to(torch.float32).to(device))
        
        # reload first linear layer
        self.l1 = torch.nn.Linear(self.n_filters_3, self.n_HU_1)
        self.l1.weight = nn.Parameter(torch.tensor(info['w_l1']).to(torch.float32).to(device))
        self.l1.bias = nn.Parameter(torch.tensor(info['b_l1']).to(torch.float32).to(device))
        
        # reload second linear layer
        self.l2 = torch.nn.Linear(self.n_HU_1, self.n_out)
        self.l2.weight = nn.Parameter(torch.tensor(info['w_l2']).to(torch.float32).to(device))
        self.l2.bias = nn.Parameter(torch.tensor(info['b_l2']).to(torch.float32).to(device))
        
    def np_forward(self, x):
        w_conv1 = self.conv1.weight.detach().cpu().numpy()
        b_conv1 = self.conv1.bias.detach().cpu().numpy()
        w_conv2 = self.conv2.weight.detach().cpu().numpy()
        b_conv2 = self.conv2.bias.detach().cpu().numpy()
        w_conv3 = self.conv3.weight.detach().cpu().numpy()
        b_conv3 = self.conv3.bias.detach().cpu().numpy()
        w_l1 = self.l1.weight.detach().cpu().numpy()
        b_l1 = self.l1.bias.detach().cpu().numpy()
        w_l2 = self.l2.weight.detach().cpu().numpy()
        b_l2 = self.l2.bias.detach().cpu().numpy()
        
        if len(x.shape) == 3:
            x = np.expand_dims(x, 0)
        
        conv1_out_t_pre_act = np.zeros((np.size(x, 0), self.n_filters_1, self.n_F+1-self.filter_area[0], 
                                        self.stim_size+1-self.filter_area[1]))
        for clip in range(np.size(x, 0)):
            for filt in range(self.n_filters_1):
                for f in range(self.filter_area[0]-1, self.n_F):
                    for t in range(self.filter_area[1]-1, self.stim_size):
                        in_x = x[clip, 0, f-(self.filter_area[0]-1):f+1, t-(self.filter_area[1]-1):t+1]
                        conv_prod = np.sum(np.multiply(w_conv1[filt, 0], in_x))
                       
                        conv1_out_t_pre_act[clip, filt, f-(self.filter_area[0]-1), t-(self.filter_area[1]-1)] = conv_prod + \
                            b_conv1[filt]
        
        conv1_out_t = 1/(1 + np.exp(-conv1_out_t_pre_act))
        
        
        n_F_conv1 = np.size(conv1_out_t, 2)
        t_size_conv1 = np.size(conv1_out_t, 3)
        
        torch_x = torch.tensor(conv1_out_t).to(torch.float32).to(device)
        conv2_out_t_pre_act = self.conv2(torch_x).detach().cpu().numpy()
        conv2_out_t = 1/(1 + np.exp(-conv2_out_t_pre_act))
        
             
        n_F_conv2 = np.size(conv2_out_t, 2)
        t_size_conv2 = np.size(conv2_out_t, 3)
        
        torch_x = torch.tensor(conv2_out_t).to(torch.float32).to(device)
        conv3_out_t_pre_act = self.conv3(torch_x).detach().cpu().numpy()
        conv3_out_t = 1/(1 + np.exp(-conv3_out_t_pre_act))
        
        
        h_pre_act_trace = np.zeros((np.size(conv3_out_t, 0), self.n_HU_1, np.size(conv3_out_t, -1)))
        h_trace = np.zeros((np.size(conv3_out_t, 0), self.n_HU_1, np.size(conv3_out_t, -1)))
        
        o_pre_act_trace = np.zeros((np.size(conv3_out_t, 0), self.n_out, np.size(conv3_out_t, -1)))
        o_trace = np.zeros((np.size(conv3_out_t, 0), self.n_out, np.size(conv3_out_t, -1)))

        for clip in range(np.size(x, 0)):
            x_in = np.reshape(np.transpose(conv3_out_t[clip], (1, 0, 2)), \
                              (conv3_out_t[clip].shape[0]*conv3_out_t[clip].shape[1], -1))

            h_t_pre_act = np.dot(w_l1, x_in) + b_l1[:, None]
            h_t = 1/(1 + np.exp(-h_t_pre_act))
            
            o_t_pre_act = np.dot(w_l2, h_t) + b_l2[:, None]
            o_t = 1/(1 + np.exp(-o_t_pre_act))
            
            h_pre_act_trace[clip] = h_t_pre_act
            h_trace[clip] = h_t
            o_pre_act_trace[clip] = o_t_pre_act
            o_trace[clip] = o_t
                
        traces = {'conv1_out_pre_act': conv1_out_t_pre_act, 'conv1_out': conv1_out_t, 
                  'conv2_out_pre_act': conv2_out_t_pre_act, 'conv2_out': conv2_out_t, 
                  'conv3_out_pre_act': conv3_out_t_pre_act, 'conv3_out': conv3_out_t, 'h_pre_act': h_pre_act_trace, 
                  'h': h_trace, 'o_pre_act': o_pre_act_trace, 'o': o_trace}
                
        return traces
    
    def np_forward_loop(self, inputs):
        inputs = torch.squeeze(inputs[:, -1, :])
        
        if inputs.size(1) > self.stim_size:
            inputs = inputs.view(self.n_F, -1, self.stim_size) # reshape to (f, n, t)
            inputs = inputs.permute(1, 0, 2) # permute dimensions to (n, f, t)
        
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(0) # add dimension (channels) at axis 0 -> (channels, f, t)
        elif len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1) # add dimension (channels) at axis 1 -> (n, channels, f, t)
        
        inputs = inputs.detach().cpu().numpy()
        
        traces = self.np_forward(inputs)

        return traces
    
    def split_sequences(self, seq1, seq2, len1, filter_len, n_splits):
        len2 = len1 - 3*(filter_len - 1)
        
        seq1_splits = torch.zeros(n_splits, np.size(seq1, 0), len1).to(device)
        seq2_splits = torch.zeros(np.size(seq2, 0), n_splits, len2).to(device)

        # Calculate starting index for seq1 to align with seq2
        start_idx1 = 0

        for i in range(n_splits):
            # Calculate ending indices ensuring correct alignment
            end_idx1 = start_idx1 + len1
            end_idx2 = end_idx1 - 3*(filter_len - 1)
            start_idx2 = end_idx2 - len2
            
            # Extract sub-sequences
            sub_seq1 = seq1[:, start_idx1:end_idx1]
            sub_seq2 = seq2[:, start_idx2:end_idx2]

            # Append to lists if lengths are correct
            if sub_seq1.size(1) == 132 and sub_seq2.size(1) == 72:
                seq1_splits[i] = sub_seq1
                seq2_splits[:, i] = sub_seq2

            start_idx1 = end_idx1
            start_idx2 = end_idx2

        return seq1_splits, seq2_splits
