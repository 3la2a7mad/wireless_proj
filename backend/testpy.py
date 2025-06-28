import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv() # This reads your GEMINI_API_KEY from the .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # This gets the value

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY is not set in .env. Please check your .env file.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY) # This tells the library to use your key
        print("\n--- Listing models that support 'generateContent' for your API key ---")
        found_gemini_pro = False
        for m in genai.list_models(): # This makes the actual API call to list models
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}") # This prints the name of each supported model
                if m.name == 'models/gemini-pro':
                    found_gemini_pro = True
        if not found_gemini_pro:
            print("\n!!! WARNING: 'models/gemini-pro' was NOT found in the list of supported models for generateContent. !!!")
            print("This means your current API key might not have access to it, or it's not available in your region.")
            print("You may need to try a different model from the list above, or generate a new API key/check your Google Cloud project settings.")
    except Exception as e:
        print(f"An error occurred while trying to list models: {e}")
        print("This often indicates a fundamental issue with the API key itself or a network problem preventing communication with Google's servers.")