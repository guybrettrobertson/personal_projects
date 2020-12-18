############################## WhatsApp Chat Bot ##############################
# This module reads and processes the chat data, builds the chat bot.
###############################################################################

# Import dependencies
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.doc2vec import Doc2Vec


# ChatBot class
class ChatBot:
    
    def __init__(self, my_name, friend_name, chat_file_path, 
                 remove_stop_words=False):
        ''' Initialise the chat bot; load, process and tag the data.
        
        Parameters
        ----------
        my_name: str
            Name of the user. Must be the name found in the user's WhatsApp 
            chat data.
        friend_name: str
            Name of the user's friend.
        chat_file_path: str
            File path of the chat data data.
        remove_stop_words = Bool
            Option to remove stop words from the chat data. Default is False 
            because stop words tend to be important for a lot of casual
            conversation.
            
        Returns
        -------
        Nil.
        '''
        
        assert type(my_name) == str, 'my_name must be a string.'
        assert type(friend_name) == str, 'friend_name must be a string.'
        assert type(my_name) == str, 'my_name must be a string.'
        
        # The name of the person who has downloaded the chat data
        self.my_name = my_name
        # The name of the other person
        self.friend_name = friend_name
        # Option to remove stop words from the data
        self.remove_stop_words = remove_stop_words
        
        # Load the raw chat data
        chat = open(chat_file_path)
        chat_text = chat.read()
        raw_chat = chat_text.splitlines()
        
        # List of messages from me
        self.me_chat = []
        # Corresponding list of responses to my messages
        self.friend_chat = []
        
        # The first element denotes the name of the sender and the second 
        # denotes the message sent
        previous_row = [None, None]

        # Iterate through all messages
        n = len(raw_chat)
        for i in range(n):
            row = raw_chat[i]
            # Check that the row is not empty
            if len(row) != 0:
                # Check that the row is valid, i.e. excludes picture messages, etc.
                # Valid rows start with a '['
                if row[0] == '[':
                    # Remove time stamp in row
                    row = row.split('] ')[1]
                    # Split the remaining string into the name of the sender,
                    # and the message
                    row = row.split(': ')[0:2]
                    # Only keep the last message in a string of messages from me, 
                    # and my friend's first response to this
                    if previous_row[0] == my_name and row[0] != my_name:
                        self.me_chat.append(previous_row[1])
                        self.friend_chat.append(row[1])
                    previous_row = row
        
        # Option to remove stop words
        if self.remove_stop_words == True:
            stop_words = stopwords.words('english')
        
        # The tagged message data
        self.tagged_data = []
        
        # Create tagged documents for each message from me
        n = len(self.me_chat)
        for i in range(n):
            message = word_tokenize(self.me_chat[i].lower())
            if self.remove_stop_words == True:
                tagged_message = [word for word in message if word not in stop_words]
            else:
                tagged_message = message
            self.tagged_data.append(TaggedDocument(words=tagged_message, tags = [i]))

    def train(self, max_epochs=100, vec_size=100, alpha=0.025):
        ''' Build and train the chat bot model.
        
        Parameters
        ----------
        max_epochs: int
            Maximum number of epochs iterated in training the model.
        vec_size: int
            Size of the vector used to model the tagged messages.
        alpha: float32
            The initial learning rate.
        
        Returns
        -------
        Nil.
        '''
        # Instantiate the Doc2Vec model
        self.model = Doc2Vec(vector_size=vec_size,
                             alpha=alpha, 
                             min_alpha=0.00025,
                             min_count=1,
                             dm =1)
        
        # Build the vocab using the tagged data
        self.model.build_vocab(self.tagged_data)
        
        # Iterate through each training epoch
        for epoch in range(max_epochs):
            # Output a message to indicate progress
            if epoch % 10 == 0:
                print('Iteration: ' +  str(epoch) + ' / ' + str(max_epochs))
            # Train the model
            self.model.train(self.tagged_data,
                             total_examples=self.model.corpus_count,
                             epochs=self.model.epochs)
            # Decrease the learning rate
            self.model.alpha -= 0.0002
            # Fix the learning rate, no decay
            self.model.min_alpha = self.model.alpha
        
        model_file_name = self.friend_name + "_doc2vec.model"
        self.model.save(model_file_name)
        self.model = Doc2Vec.load(model_file_name)
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, 
                                                  keep_inference=True)
        print('Iteration: ' +  str(max_epochs) + ' / ' + str(max_epochs))
        print("Model Saved")
        
    def message(self, message):
        ''' Prints a response to the input message.
        
        Parameters
        ----------
        message: str
            Input message.
        
        Returns
        -------
        str
            response message
        '''
        
        # Make the message lower case and tokenize it
        message = word_tokenize(message.lower())
        
        # Option to remove stop words
        if self.remove_stop_words == True:
            message = [word for word in message if word not in stop_words]
            
        # Infer the message's vector
        message_vector = self.model.infer_vector(message, epochs=1000)
        
        # Find the most similar message to the message given
        similar_message = self.model.docvecs.most_similar([message_vector])[0][0]
        print(self.friend_name + ': ' + self.friend_chat[similar_message])
        return(self.friend_name + ': ' + self.friend_chat[similar_message])