
class PyomoSubProblems(SubProblemsBase):
    """Container class for managing all sub-problems
    
    
    """
    def __init__(self, sub_models, cuts, block_id, settings):

        self.surrogate_model = SurrogateModel

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
class SurrogateModel:
    """Class for implementation, training of the NN
        as instance to the pyomosubproblem
        write test
    """
    def __init__(self,block_id, tdata, neurons, max_iter = 200):
        """Constructor 
        """
        self.data = tdata[block_id]
        self.max_iter = max_iter
        self.block_id = block_id

        #Setting up the neural network
        self.NN = MLPClassifier(hidden_layer_size = (neurons, neurons),
        alpha = 1e-5,
        activation = 'relu',
        batch_size = '',
        solver = 'sgd',
        max_iter = self.max_iter)

    def init_training(self):
        '''initial training of the NN
        Scale data with StandardScaler
        '''
        #Scaling directions (input)
        self.x_scaled = StandardScaler().fit(self.data[block_id][0])
        #Scaling inner points (predictions)
        self.y_scaled = StandardScaler().fit(self.data[block_id][1])
        X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, random_state = 1)

        clf = self.NN
        clf.fit(X_train, y_train)
        
    def predict(self, direction):
        return point

