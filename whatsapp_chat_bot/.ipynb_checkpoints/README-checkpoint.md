# Whatsapp chat bot

## Summary

WhatsApp allows you to download the message history of any conversation. In this project, I have built a model that can take any WhatsApp message history and turn it into a chat bot. As the chat bot is based on the messages of the friend the user has been messaging, the chat bot's personality is based on that of the user's friend.

The user uploads a WhatsApp message history, and the chat bot learns from this dataset. The user then inputs a message, and the model finds a message sent by the user in the dataset which is most similar to the input message. The model then selects the response corresponding to this most similar message from the dataset. The response is then given by the chat bot.

## Methodology

First, the model loads and processes the conversation history, which has been downloaded by the user directly from WhatsApp as a .txt file. Conversations tend to have groups of messages from each person before the other person responds. As a simplification, the model extracts the last message of a group sent by the user, and the first message received in response to this. This gives a pair of messages: a message sent by the user and a response from the user's friend. So at this stage the model has an array of messages and an array of corresponding responses.

In processing the data, the model has an option to remove stop words. Stop words are typically removed from data used for Natural Language Processing because they are very common and have little specialised meaning. However, I have found that removing stop words in the chat bot worsens the results. This is likely to be because small talk contains a lot of stop words. For example, 'how', 'are', and 'you' are all stop words. By default, the chat bot therefore does not remove stop words.

The chat bot also makes all of the chat data lower case, and by doing so assumes there is no important distinction between upper and lower case letters in the WhatsApp conversation.

Once the chat history has been converted into a list of messages from the user and responses to the user. The messages from the user are tagged and used to train a doc2vec model which uses DM-PV (Distributed Memory version of Paragraph Vector) to convert each message into a numerical vector, where each vector corresponds to a position in 'meaning space'.

After the chat bot has been trained on the input message history, it can have a conversation with the user. The user must input a message, and the chat bot then takes that message and converts it into a vector. The chat bot then finds the vector in the meaning space which closest to the new message vector. In other words, it finds the message that has the closest meaning to the new message. The chat bot then simply returns the response from the user's friend that corresponds to the closest message.

I have tested the chat bot on two different conversations here, using three types of input messages:

- Messages taken from the input data, which we would expect to result in the same response that was given in the input data);
- Messages similar but not identical to messages in the input data, which we would expect to give some reasonable responses; and
- Totally new messages, which we would not expect sensible responses, given how simplistic the chat bot is.