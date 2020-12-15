# WhatsApp chat bot

WhatsApp allows you to download the message history of any conversation. In this project, I have built a model that can take any WhatsApp message history and turn it into a chat bot. As the chat bot is based on the messages of the friend the user has been messaging, the chat bot's personality is based on that of your friend.

The user uploads a WhatsApp message history, and the chat bot learns from this dataset. The user then inputs a message, and the model finds a message sent by the user in the dataset which is most similar to the input message. The model then selects the response corresponding to this most similar message from the dataset. The response is then given by the chat bot.

I have used a Doc2Vec model with the nltk and gensim packages (in Python) to create the chat bot.