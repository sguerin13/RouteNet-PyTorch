import torch
import torch.nn as nn
import torch.nn.functional as F

'''using the implementation from RouteNet'''
class RouteNet(nn.Module):

    def __init__(self):
        super(RouteNet,self).__init__()

        ### Architecture ###
        # for gru need to pay attention to if input is of size:
        # (batch, seq_len, feature size) or seq_len, batch, feature size
        # if sequence length is variable
        # may need to pad the sequence
        self.link_state_dim = 512
        self.path_state_dim = 512
        self.readout_dim = 256
        self.output_units = 1
        self.T = 8
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        hSize  = 512
        nLayer = 1

        # RNN for link hidden state update
        self.l_U = nn.GRU(input_size = self.link_state_dim,
                          hidden_size = hSize,
                          num_layers = nLayer,
                          batch_first=True)
        
        # RNN for path hidden state update
        self.p_U = nn.GRU(input_size = self.path_state_dim,
                          hidden_size = hSize,
                          num_layers = nLayer,
                          batch_first=True)
        
        # Readout layers
        self.readOut_Delay = nn.ModuleDict({'r1': nn.Linear(hSize,self.readout_dim),
                                            'r2': nn.Linear(self.readout_dim,int(self.readout_dim/2)),
                                            'r3': nn.Linear(int(self.readout_dim/2),self.output_units)
                                            })
        
        self.readOut_Jitter = nn.ModuleDict({'r1': nn.Linear(hSize,self.readout_dim),
                                             'r2': nn.Linear(self.readout_dim,int(self.readout_dim/2)),
                                             'r3': nn.Linear(int(self.readout_dim/2),self.output_units)
                                              })


    def forward(self,x):
        '''
        The sample network data is preprocessed in the Dataset class and mapped to the following format,
        representing the graph as a list of links and paths:

        The variables 'links', 'paths', and 'seqs' are flattened representations of the
        information in the path list.                                            (P_0), (P_1)
            - The original path list is a nested list of the links in the path: [[0,1],[1,2,3], ...]

        The input data has the following format:
            links: Represents the links on the paths:                                             [0,1,1,2,3,...]
            paths: Represents the path index for each entry above:                                [0,0,1,1,1,...]
            seqs:  Represents the sequence # for each link in its respective path for each entry: [0,1,0,1,2,...]

            Link capacity: is a vector of the link capacities for each link in the network - shape (n_links,1) - this is the input feature for the link nodes
            Bandwidth: is a vector of average bandwidth for each path in the network - shape (n_paths,1) - this is the input feature for the path nodes

        '''
        links_indices = x['links']                   
        path_indices = x['paths']            
        seq_indices = x['sequences']
        link_cap = x['link_capacity']
        bandwidth = x['bandwith']

        # TODO: why the minus 1? - I think it is superfluous
        # link hidden state matrix (n_links x h_state_dim)
        link_h_state_shape = (x['n_links'][0], self.link_state_dim-1)

        # path hidden state matrix (n_links x h_state_dim)
        path_h_state_shape = (x['n_paths'],self.path_state_dim-1)
        # path_h_state = torch.cat((bandwidth,torch.zeros(path_h_state_shape).to(self.device)), axis=1)
        
        # prepare input for path update RNN
        max_seq_len = torch.max(seq_indices)
        path_rnn_input_shape = (x['n_paths'],max_seq_len+1,self.link_state_dim)
        
        #stack the paths and sequences
        ids = torch.stack((path_indices,seq_indices),axis=1)
        ids = torch.squeeze(ids,2)           
        p_ind = ids[:,0]
        s_ind = ids[:,1]

        # Instead of using a double for-loop to update the link states from the path states, we'll build a matrix and do it in one pass
        # - Create indices for gathering the link hidden states, shape: (n_links,link_state_dim)
        # See line (150ish) #TODO
        indices = torch.zeros(len(links_indices),self.link_state_dim).to(self.device)
        for i in range(len(links_indices)):
            link_id = links_indices[i]
            indices[i,:] = link_id
            
        # variable dictionaries for forward pass - keep track of states to use for gradient propagation
        path_rnn_input = {}
        path_h_state = {}
        path_h_state_seq = {}
        link_h_state = {}
        h_links_on_paths = {}
        link_message = {}
        aggregated_link_message = {}

        
        for t in range(self.T):
            ############# set up the matrices and variables for each pass through #################
            
            ########## PATH VARIABLES ###########
            
            # input to the path rnn layer P_u
            path_rnn_input[t] = torch.zeros(path_rnn_input_shape).to(self.device)
            
            if (t > 0):  # for non leaf variables, we need to propagate the gradient back
                path_rnn_input[t].requires_grad = True
            
            # path hidden state output from P_U, initialized with just bandwidth at T_0, else copy from previous loop
            if (t==0):
                path_h_state[t] = torch.cat((bandwidth,torch.zeros(path_h_state_shape).to(self.device)), axis=1)
            else:
                path_h_state[t] = path_h_state[t-1]
            
            
            # path_hidden state sequence from P_U, used to update links
            path_h_state_seq_key = 'path_h_states_seq_' + str(t)
            
            
            ########## LINK VARIABLES ###########
            
            # vector to store the link_hidden states
            if (t == 0):
                # create hidden state matrix for links and initialize with first column as link capacity
                link_h_state[t] = torch.cat((link_cap,torch.zeros(link_h_state_shape).to(self.device)),1)
            else:
                # copy hidden state value for next pass through
                link_h_state[t] = link_h_state[t-1]
                
            # matrix storing the hidden states of links on paths
            # i.e. the hidden state of all links in the x['links'] list
            h_links_on_paths[t] = torch.gather(link_h_state[t],0,indices.long())
            
            #link messages extracted from the path hidden state sequence output from P_U
            link_message_key = 'link_messages_' + str(t)

            # container for the link messages that are extracted from path rnn hidden states
            aggregated_link_message[t] = torch.zeros((x['n_links'],self.link_state_dim),requires_grad=True).to(self.device)
            
            ########################################################################################
            
            
            ################################## DO THE MESSAGE PASSING ##############################
            
            # prepare input for path RNN
            path_rnn_input[t] = path_rnn_input[t].index_put(indices = [p_ind,s_ind],
                                                                      values =h_links_on_paths[t])
            # pass through the path RNN
            path_h_state_seq[t], path_h_state[t] = self.p_U(path_rnn_input[t],
                                                                      torch.unsqueeze(path_h_state[t],
                                                                      0))
            # reformat
            path_h_state[t] = path_h_state[t].squeeze(0)
            
            # extract link messages from the path RNN sequence output
            # equivalent to tf.gather_nd
            link_message[t] = path_h_state_seq[t][p_ind,s_ind,:]
           
            # aggregate the link messages
            aggregated_link_message[t] = aggregated_link_message[t].index_put([links_indices.squeeze(1)],
                                                                           link_message[t],
                                                                           accumulate=True)
            # update the state of the links by passing through link 
            _, link_h_state[t] = self.l_U(torch.unsqueeze(aggregated_link_message[t],1),
                                               torch.unsqueeze(link_h_state[t].squeeze(0),0))
            # reformat
            link_h_state[t] = link_h_state[t].squeeze(0)
            
            ##########################################################################################

        # readout from the paths
        d,j = self.readout(path_h_state[t])
        return d,j


    def readout(self,path_state):
        d = F.relu(self.readOut_Delay['r1'](path_state))
        d = F.relu(self.readOut_Delay['r2'](d))
        d = self.readOut_Delay['r3'](d)

        j = F.relu(self.readOut_Jitter['r1'](path_state))
        j = F.relu(self.readOut_Jitter['r2'](j))
        j = self.readOut_Jitter['r3'](j)
        
        return (d,j)