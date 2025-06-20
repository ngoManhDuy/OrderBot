import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CoffeeChatbot:
    def __init__(self, api_key: str = None):
        """
        Initialize the chatbot with OpenAI API key
        """
        # Set API key from parameter or environment variable
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY environment variable
            self.client = OpenAI()
        
        # Conversation history
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": """You are a helpful coffee shop assistant chatbot. 
                You help customers with their orders and answer questions about coffee, 
                menu items, and the coffee shop. Be friendly, knowledgeable, and concise."""
            }
        ]
        
        # Model configuration
        self.model = "gpt-4o"  # Latest GPT-4 model
        self.max_tokens = 1000
        self.temperature = 0.7
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
        
        # Keep conversation history manageable (last 20 messages)
        if len(self.messages) > 21:  # 1 system + 20 conversation messages
            self.messages = [self.messages[0]] + self.messages[-20:]
    
    def get_response(self, user_input: str) -> str:
        """
        Get chatbot response for user input
        """
        try:
            # Add user message to history
            self.add_message("user", user_input)
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            
            # Extract response content
            bot_response = response.choices[0].message.content
            
            # Add bot response to history
            self.add_message("assistant", bot_response)
            
            return bot_response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_streaming_response(self, user_input: str):
        """
        Get streaming response (yields chunks as they arrive)
        """
        try:
            # Add user message to history
            self.add_message("user", user_input)
            
            # Make streaming API call
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            
            full_response = ""
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Add complete response to history
            self.add_message("assistant", full_response)
            
        except Exception as e:
            yield f"Sorry, I encountered an error: {str(e)}"
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.messages = [self.messages[0]]  # Keep only system message
    
    def set_system_prompt(self, prompt: str):
        """Update the system prompt"""
        self.messages[0]["content"] = prompt


def main():
    """
    Main function to run the chatbot
    """
    print("☕ Coffee Shop Chatbot")
    print("=" * 40)
    print("Type 'quit' to exit, 'reset' to clear conversation, 'stream' to toggle streaming mode")
    print()
    
    # Initialize chatbot
    try:
        bot = CoffeeChatbot()
        streaming_mode = False
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Thanks for chatting! Have a great day! ☕")
                break
            
            if user_input.lower() == 'reset':
                bot.reset_conversation()
                print("Conversation reset!")
                continue
            
            if user_input.lower() == 'stream':
                streaming_mode = not streaming_mode
                print(f"Streaming mode: {'ON' if streaming_mode else 'OFF'}")
                continue
            
            if not user_input:
                continue
            
            print("Bot: ", end="")
            
            if streaming_mode:
                # Streaming response
                for chunk in bot.get_streaming_response(user_input):
                    print(chunk, end="", flush=True)
                print()  # New line after streaming
            else:
                # Regular response
                response = bot.get_response(user_input)
                print(response)
            
            print()
    
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable")


# Example usage with custom configuration
def example_usage():

    bot = CoffeeChatbot()
    
    # Customize system prompt
    bot.set_system_prompt("""
    Bạn là một nhân viên order của quán cà phê tên Capuchino Assasino, nhiệm vụ của bạn là trợ giúp khách 
    hàng order đồ uống một cách thật cẩn thận và chi tiết cho khách hàng. Lưu ý là bạn hãy cố gắng giao tiếp 
    với khách hàng bằng cả tiếng việt và tiếng anh, tùy thuộc vào ngôn ngữ chính mà khách hàng sử dụng. Nhiều lúc yêu cầu
    của khách hàng có thể không rõ ràng, vì vậy bạn cần làm rõ ràng yêu cầu của khách nhé. 
    """)
    
    # Example conversation
    responses = [
        "Hello! What's your best espresso?",
        "Can you recommend something sweet?",
        "What's the difference between cappuccino and latte?"
    ]
    
    for question in responses:
        print(f"Customer: {question}")
        response = bot.get_response(question)
        print(f"Mario: {response}\n")


if __name__ == "__main__":
    # Uncomment to run example
    # example_usage()
    
    # Run interactive chatbot
    main()