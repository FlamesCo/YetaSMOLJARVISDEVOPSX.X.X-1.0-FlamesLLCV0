#!/usr/bin/env python3

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

# ASCII art of a cat
print('''
 /\_/\  
( o.o ) 
 > ^ <
''')

print("Welcome to 'YET A SMOL JARVIS 1.0'")

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()

if torch.cuda.is_available():
    model.to('cuda')

while True:
    print("\nOptions:\n1. Generate Text\n2. Exit")
    option = input("Please choose an option: ")

    if option == '1':
        # Request a prompt from the user
        prompt = input("Please enter your prompt: ")

        # Generate a response using the loaded model
        inputs = tokenizer.encode(prompt, return_tensors='pt')

        if torch.cuda.is_available():
            inputs = inputs.to('cuda')

        outputs = model.generate(inputs, max_length=150, do_sample=True, temperature=0.7)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("\nHere's what I came up with:\n")
        print(generated_text)
        
        # Ask user if they want to store the generated text in a file
        store_option = input("\nDo you want to store this text in a file? (y/n): ")
        if store_option.lower() == 'y':
            with open('generated_text.txt', 'a') as file:
                file.write(generated_text + '\n')
            print("Text has been stored in 'generated_text.txt' file")

        # Ask user if they want to use Mac's text-to-speech feature
        tts_option = input("\nDo you want to use Mac's text-to-speech feature to read the text out loud? (y/n): ")
        if tts_option.lower() == 'y':
            os.system(f"say {generated_text}")
    elif option == '2':
        print("\nThank you for using 'YET A SMOL JARVIS 1.0'. Goodbye!")
        break
    else:
        print("\nInvalid option. Please choose either 1 or 2.")
