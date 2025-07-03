#!/usr/bin/env python3
import gradio as gr
import os
import sys
import threading
import time
from typing import Tuple, Optional

# Add paths for imports
sys.path.append('voice_handle')
sys.path.append('ordering_chatbot')

from voice_handle.stt_whisper import STT_module
from ordering_chatbot.OrderBot import OrderBot

class VoiceOrderingUI:
    """Simple UI for Highland Coffee Voice Ordering System"""
    
    def __init__(self):
        self.stt = None
        self.orderbot = None
        self.conversation_text = "Start speaking to place your order.\n\n"
        self.is_processing = False
        self.initialize_system()
        self.start_voice_loop()
    
    def initialize_system(self):
        """Initialize STT and OrderBot systems"""
        try:
            print("🔄 Initializing Speech-to-Text module...")
            self.stt = STT_module(
                model_name="openai/whisper-medium",
                enable_denoising=True,
                status_callback=self.print_status
            )
            
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
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            raise
    
    def print_status(self, message):
        """Status callback for STT module"""
        print(f"[STT] {message}")
    
    def start_voice_loop(self):
        """Start the automatic voice detection loop"""
        def voice_loop():
            while True:
                try:
                    if not self.is_processing:
                        self.process_voice()
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error in voice loop: {e}")
                    time.sleep(1)
        
        # Start voice loop in background thread
        voice_thread = threading.Thread(target=voice_loop, daemon=True)
        voice_thread.start()
    
    def format_user_message(self, message: str) -> str:
        """Format user message for left alignment"""
        # Create a box-like format for user messages
        border = "┌" + "─" * 50 + "┐"
        footer = "└" + "─" * 50 + "┘"
        lines = []
        
        # Add header
        lines.append("👤 You")
        lines.append(border)
        
        # Split message into lines that fit within the box
        words = message.split()
        current_line = ""
        for word in words:
            if len(current_line + word + " ") <= 48:  # Leave space for borders
                current_line += word + " "
            else:
                if current_line:
                    lines.append("│ " + current_line.ljust(48) + " │")
                current_line = word + " "
        
        # Add the last line if there's content
        if current_line:
            lines.append("│ " + current_line.ljust(48) + " │")
        
        lines.append(footer)
        lines.append("")  # Empty line for spacing
        
        return "\n".join(lines)
    
    def format_bot_message(self, message: str) -> str:
        """Format bot message for right alignment"""
        # Create a box-like format for bot messages, right-aligned
        border = "┌" + "─" * 50 + "┐"
        footer = "└" + "─" * 50 + "┘"
        lines = []
        
        # Right alignment spacing - moved much further right
        right_space = " " * 60
        
        # Add header
        lines.append(right_space + "🤖 Bot")
        lines.append(right_space + border)
        
        # Split message into lines that fit within the box
        words = message.split()
        current_line = ""
        for word in words:
            if len(current_line + word + " ") <= 48:  # Leave space for borders
                current_line += word + " "
            else:
                if current_line:
                    lines.append(right_space + "│ " + current_line.ljust(48) + " │")
                current_line = word + " "
        
        # Add the last line if there's content
        if current_line:
            lines.append(right_space + "│ " + current_line.ljust(48) + " │")
        
        lines.append(right_space + footer)
        lines.append("")  # Empty line for spacing
        
        return "\n".join(lines)
    
    def format_system_message(self, message: str) -> str:
        """Format system message for center alignment"""
        right_space = " " * 25
        return f"{right_space}🔔 {message}\n\n"
    
    def process_voice(self):
        """Process voice input automatically"""
        if self.is_processing:
            return
        
        try:
            self.is_processing = True
            
            # Get voice input using VAD
            raw_text, corrected_text = self.stt.process_single_recording(max_duration=30)
            
            if raw_text is not None:
                print(f"[UI] User said: {corrected_text}")
                
                # Add user input to conversation (left-aligned)
                self.conversation_text += self.format_user_message(corrected_text)
                
                # Check for exit commands
                if corrected_text.lower().strip() in ['quit', 'exit', 'thoát', 'kết thúc', 'bye', 'goodbye']:
                    # Add bot response (right-aligned)
                    self.conversation_text += self.format_bot_message("Thank you for visiting Highland Coffee! Goodbye!")
                    return
                
                # Process with OrderBot
                bot_response = self.orderbot.process_message(corrected_text)
                print(f"[UI] Bot response: {bot_response}")
                
                # Add bot response to conversation (right-aligned)
                self.conversation_text += self.format_bot_message(bot_response)
                
                # Check if order is complete
                if self.orderbot.is_order_complete():
                    self.conversation_text += self.format_system_message("✅ ORDER COMPLETED!")
                    self.conversation_text += self.format_system_message("📄 Bill has been generated successfully.")
                    self.conversation_text += self.format_system_message("Cảm ơn bạn đã đến với Highland Coffee! Thank you!")
                    
                    # Reset for next order
                    time.sleep(3)  # Show completion message for 3 seconds
                    self.reset_conversation()
        
        except Exception as e:
            print(f"Error in voice processing: {e}")
            # Error message (right-aligned)
            self.conversation_text += self.format_system_message(f"❌ Error: {e}")
        
        finally:
            self.is_processing = False
    
    def reset_conversation(self):
        """Reset conversation and order for next customer"""
        print("[UI] Resetting for next customer...")
        self.conversation_text = "Welcome to OrderBot! Start speaking to place your order.\n\n"
        if self.orderbot:
            self.orderbot.reset_conversation()
    
    def get_conversation(self):
        """Get current conversation text"""
        return self.conversation_text
    
    def get_order_status(self):
        """Get current order status"""
        if not self.orderbot or not self.orderbot.current_order:
            return "📝 Đơn hàng trống / Order is empty"
        
        try:
            return self.orderbot.show_current_order()
        except Exception as e:
            return f"❌ Error getting order: {e}"
    
    def get_status(self):
        """Get current system status"""
        if self.is_processing:
            return "🎤 Processing your speech..."
        else:
            return "🎤 Ready - Speak your order"
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        css = """
        .menu-container img {
            width: auto !important;
            height: auto !important;
            max-width: 100% !important;
            object-fit: contain !important;
        }
        .conversation-box textarea {
            font-family: monospace !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
        }
        """
        
        with gr.Blocks(css=css, title="OderBot: Voice Ordering") as interface:
            gr.Markdown("# OderBot: Voice Ordering System")
            
            with gr.Row():
                # Left side: Chat (50% width)
                with gr.Column(scale=1):
                    gr.Markdown("## Conversation")
                    
                    status_display = gr.Textbox(
                        value="🎤 Ready - Speak your order",
                        label="Status",
                        interactive=False,
                        max_lines=1
                    )
                    
                    conversation_display = gr.Textbox(
                        value="Start speaking to place your order.\n\n",
                        label="Chat",
                        lines=20,
                        max_lines=25,
                        interactive=False,
                        elem_classes=["conversation-box"]
                    )
                
                # Right side: Menu and Order (50% width)
                with gr.Column(scale=1):
                    gr.Markdown("## 📋 Menu")
                    menu_image = gr.Image(
                        value="menu.png",
                        label="Highland Coffee Menu",
                        interactive=False,
                        container=False,
                        show_label=False,
                        show_download_button=False,
                        show_share_button=False
                    )
                    
                    gr.Markdown("## 📋 Current Order")
                    order_display = gr.Textbox(
                        value="📝 Đơn hàng trống / Order is empty",
                        label="Order Details",
                        lines=8,
                        max_lines=12,
                        interactive=False
                    )
            
            # Update function
            def update_display():
                return (
                    self.get_conversation(),
                    self.get_order_status(),
                    self.get_status()
                )
            
            # Auto-update the display every 1 second using a timer
            timer = gr.Timer(1.0)
            timer.tick(
                fn=update_display,
                outputs=[conversation_display, order_display, status_display]
            )
        
        return interface

def main():
    """Main entry point"""
    try:
        print("🚀 Starting Highland Coffee Voice Ordering UI...")
        
        # Create UI system
        ui_system = VoiceOrderingUI()
        
        # Create and launch interface
        interface = ui_system.create_interface()
        
        # Launch
        interface.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Error starting UI: {e}")
        print("💡 Please check your dependencies and try again")

if __name__ == "__main__":
    main() 