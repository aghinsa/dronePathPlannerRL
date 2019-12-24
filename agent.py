import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten
from keras.optimizers import Adam
from autologging import traced

@traced
class ReplayMemory(object):
    """
    Replay buffer
    """
    def __init__(self,size_of_buffer,sample_shape,history_length):
        self._pos = 0 #index to buffer
        self._count = 0
        self._max_size = size_of_buffer
        self._history_length = max(1,history_length)
        self._state_shape = sample_shape
        
        self._states = np.zeros([size_of_buffer]+sample_shape ,dtype=np.float32)
        self._actions = np.zeros(size_of_buffer,dtype=np.uint8)
        self._rewards = np.zeros(size_of_buffer,dtype=np.float32)
        self._isdone = np.zeros(size_of_buffer,dtype=np.float32)

    def __len__(self):
        return self._count
    
    def append(self,state,action,reward,isdone):
        """
        Appends the given (s,a,r,done) to buffer
        """
        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._isdone[self._pos] = isdone

        self._count = max(self._count,self._pos + 1) # so that behaviour is same when buffer is full
        self._pos = (self._pos + 1) % self._max_size

    def sample(self,size):
        """
         returns [INT] of length size ,ie indexes to access buffer 
                (use function get_item to access the indexes)
        """

        idxs = []
        pos,count,max_size = self._pos,self._count,self._max_size
        history_length = self._history_length
        while(len(idxs) < size):
            i = np.random.randint(pos,count)
            if i not in idxs :
                if ( (i<=pos) or (i>pos and i < pos+history_length) ) :
                    idxs.append(i)
        return idxs
    
    def get_state(self,index):
        """
            Returns the spedified state with the replay memory.
            A state consists of the last 'history_length' perceptions

            returns : ndarray [history_length,state_shape]
        """
        if index>self._count-1:
            raise IndexError('Out of bounds of buffer')
        history_length = self._history_length
        index %= self._count
        if index>=history_length:
            return self._states[index-history_length : index, ...]
        else:
            idxs = np.arange(index-history_length,index)
            return self._states.take(idxs,mode="wrap",axis=0)

    def minibatch(self,size):
        """
        Return :
            tuple: Tensor [size,input_shape] ,[Int](size), [size,input_shape] ,[float](size),
                    [bool](size)
        """
        indexes = self.sample(size)
        st = np.array([self.get_state(index) for index in indexes],dtype=np.float32)
        st_1 =  np.array([self.get_state(index+1) for index in indexes],dtype=np.float32)
        a = self._actions[indexes]
        r = self._rewards[indexes]
        isdone = self._isdone[indexes]

        return st,a,st_1,r,isdone

@traced
class DQNAgent(object):
    #TODO 
    
    def __init__(self,num_actions,input_shape,learning_rate,mini_batch_size=32,
        memory_size=500000,train_after=10000,train_interval=4,target_update_interval=10000,
        history_length=4,epsilon=0.95,epsilon_decay=.995,decay_interval=3000,gamma=.99,
        target_update_tau=0.125):
        """
            num_actions : indexed from 0
        """
        self._epsilon = epsilon
        self._gamma=gamma

        self._epsilon_decay=epsilon_decay
        self._decay_interval=decay_interval
        self._num_actions = num_actions  # to check for target update itervals 
        self._input_shape = input_shape 
        self._learning_rate = learning_rate
        self._memory_size = memory_size
        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_tau=target_update_tau
        self._traget_update_interval = target_update_interval
        self._history_length=history_length
        self._mini_batch_size=mini_batch_size
        self._total_num_actions=1

        self._dqn_net = self.create_model(input_shape,num_actions,'DQNet')
        self._target_net =self.create_model(input_shape,num_actions,'TargetNet')
        self._dqn_net.compile(loss="mean_squared_error",optimizer=Adam(lr=self._learning_rate)) #todo check huber loss
        
        self._memory=ReplayMemory(memory_size,input_shape,history_length)
        
        self._episode_q_means,self._episode_rewards = [],[]


    def create_model(self,input_shape,num_actions,name):
        # approximater
        with tf.variable_scope(name):
            model=Sequential()
            model.add(Conv2D(input_shape=input_shape,filters=16,kernel_size=8,strides=4,activation='relu'))
            model.add(Conv2D(filters=32,kernel_size=4,strides=2,activation='relu'))
            model.add(Conv2D(filters=32,kernel_size=3,strides=1,activation='relu'))
            model.add(Flatten())
            model.add(Dense(256,activation='relu'))
            model.add(Dense(num_actions,activation='relu'))
        return model
    
    def act(self,state):
        self._total_num_actions+=1
        if(self._total_num_actions>self._train_after and self._total_num_actions%self._decay_interval==0):
            self._epsilon*=self._epsilon_decay
        if np.random.random()<self._epsilon:          
            action=np.random.randint(0,self._num_actions)
        else:
            action=np.argmax(self._dqn_net.predict(state)[0])
        return action
    def remember(self,state,action,reward,isdone):
        self._memory.append(state,action,reward,isdone)
    
    def computeQTargets(self,post_states,rewards,isdone):
        return tf.where(isdone,rewards,
                rewards + self._gamma * tf.reduce_max(self._target_net.predict(post_states),axis=0) )

    def train(self):
        t_num_actions = self._total_num_actions
        if(t_num_actions > self._train_after) :
            if(t_num_actions % self._train_interval == 0):
                pre_states,actions,rewards,post_states,isdone=self._memory.minibatch(self._mini_batch_size)
                print(f"Mean rewards :{np.mean(rewards)}")
                q_target = self.computeQTargets(post_states,rewards,isdone)
                self._dqn_net.fit(pre_states,q_target,epochs=1)

            if(t_num_actions%self._traget_update_interval):
                weights = self._dqn_net.get_weights()
                target_weights = self._target_net.get_weights()
                for i in range(len(target_weights)):
                    target_weights[i] = weights[i] * self._target_update_tau + target_weights[i] * (1 - self._target_update_tau)
                self._target_net.set_weights(target_weights)
        


