# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can use any model of your choice
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to chat with the model
def chat_with_bot():
    print("Chatbot: Hi! I'm a chatbot. Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")

        # If the user types 'quit', exit the loop
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        
        # Encode the user input and generate a response
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        
        # Generate the model's output
        outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the generated tokens into text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input part of the response (we only want the chatbot's response)
        chatbot_response = response[len(user_input):]
        
        print(f"Chatbot: {chatbot_response.strip()}")

# Run the chatbot
chat_with_bot()
