#!/usr/bin/env python3
"""
Highland Coffee Voice Ordering System
Integrates STT (Speech-to-Text) with OrderBot for voice-controlled ordering
"""

import os
import sys
import time

# Add paths for imports
sys.path.append('voice_handle')
sys.path.append('ordering_chatbot')

from voice_handle.stt_whisper import STT_module
from ordering_chatbot.OrderBot import OrderBot

class VoiceOrderingSystem:
    """Main system that integrates STT with OrderBot"""
    
    def __init__(self):
        print("🎤 Highland Coffee Voice Ordering System")
        print("=" * 50)
        
        # Initialize STT module with status callback
        print("🔄 Initializing Speech-to-Text module...")
        self.stt = STT_module(
            model_name="openai/whisper-medium",
            enable_denoising=True,
            status_callback=self.print_status
        )
        
        # Initialize OrderBot with correct working directory
        print("🔄 Initializing OrderBot...")
        
        # Save current directory
        original_cwd = os.getcwd()
        
        try:
            # Change to ordering_chatbot directory to ensure correct config loading
            os.chdir('ordering_chatbot')
            self.orderbot = OrderBot()
        finally:
            # Restore original directory
            os.chdir(original_cwd)
        
        print("✅ System ready!")
        print("\n📋 Instructions:")
        print("• Speak your order when prompted")
        print("• System will automatically detect when you start/stop speaking") 
        print("• Order will be processed and bill generated when complete")
        print("• Say 'quit' or 'exit' to end the session")
        print("=" * 50)
    
    def print_status(self, message):
        """Status callback for STT module"""
        print(f"[STT] {message}")
    
    def start_conversation(self):
        """Start the voice ordering conversation"""
        # Send welcome message to OrderBot
        print("\n🤖 Bot: ", end="")
        welcome_response = self.orderbot.process_message("Chào mừng khách hàng đến với Highland Coffee. Tôi có thể giúp gì cho bạn?")
        print(welcome_response)
        
        return self.voice_conversation_loop()
    
    def voice_conversation_loop(self):
        """Main conversation loop using voice input"""
        conversation_count = 0
        
        while True:
            conversation_count += 1
            print(f"\n--- Conversation {conversation_count} ---")
            
            try:
                # Get voice input from user
                print("🎤 Please speak your message...")
                raw_text, corrected_text = self.stt.process_single_recording(max_duration=30)
                
                if raw_text is None:
                    print("❌ No speech detected. Please try again.")
                    continue
                
                # Show what was heard
                print(f"👤 You said: {corrected_text}")
                
                # Check for exit commands
                if corrected_text.lower().strip() in ['quit', 'exit', 'thoát', 'kết thúc', 'bye', 'goodbye']:
                    print("👋 Thank you for visiting Highland Coffee! Goodbye!")
                    break
                
                # Process with OrderBot
                print("🤖 Bot: ", end="")
                response = self.orderbot.process_message(corrected_text)
                print(response)
                
                # Check if order is complete
                if self.orderbot.is_order_complete():
                    print("\n" + "="*50)
                    print("✅ ORDER COMPLETED!")
                    print("📄 Bill has been generated successfully.")
                    print("="*50)
                    
                    # Thank the customer and end the program
                    print("\n🤖 Bot: Cảm ơn bạn đã đến với Highland Coffee! Chúc bạn một ngày tốt lành!")
                    print("🤖 Bot: Thank you for visiting Highland Coffee! Have a great day!")
                    print("\n👋 Goodbye!")
                    break
                
            except KeyboardInterrupt:
                print("\n\n🛑 System interrupted by user")
                print("👋 Thank you for visiting Highland Coffee! Goodbye!")
                break
                
            except Exception as e:
                print(f"❌ Error occurred: {e}")
                print("🔄 Continuing with voice ordering...")
                continue
        
        return True

def main():
    """Main entry point"""
    try:
        # Create and start the voice ordering system
        system = VoiceOrderingSystem()
        
        # Start the conversation
        system.start_conversation()
        
    except KeyboardInterrupt:
        print("\n\n🛑 System shutting down...")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("💡 Please check your dependencies and try again")
        
    finally:
        print("📄 Session ended. Thank you!")

if __name__ == "__main__":
    main() 