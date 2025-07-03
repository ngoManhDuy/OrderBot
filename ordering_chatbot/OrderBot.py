import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv
from vector_store_manager import VectorStoreManager
import re

load_dotenv()
client = OpenAI()

@dataclass
class OrderItem:
    name: str
    size: str
    quantity: int
    unit_price: int
    modifications: List[str] = field(default_factory=list)
    
    @property
    def total_price(self) -> int:
        return self.quantity * self.unit_price
    
    def to_dict(self):
        return {
            "name": self.name,
            "size": self.size,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "total_price": self.total_price,
            "modifications": self.modifications
        }

class OrderBot:
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.vector_store_id = self.vector_manager.get_vector_store_id()
        self.current_order: List[OrderItem] = []
        self.conversation_history = []
        
        # Define tools for the LLM
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_menu",
                    "description": "Search the Highland Coffee menu for items, prices, and information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for menu items (e.g., 'coffee', 'latte', 'prices')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_to_order",
                    "description": "Add items to the current order",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "Item name"},
                                        "size": {"type": "string", "description": "Size (S/M/L/XL)"},
                                        "quantity": {"type": "integer", "description": "Quantity"},
                                        "unit_price": {"type": "integer", "description": "Price per unit in VND"},
                                        "modifications": {"type": "array", "items": {"type": "string"}, "description": "Special requests"}
                                    },
                                    "required": ["name", "size", "quantity", "unit_price"]
                                }
                            }
                        },
                        "required": ["items"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "show_current_order",
                    "description": "Display the current order with items and total",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clear_order",
                    "description": "Clear all items from the current order",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_bill",
                    "description": "Generate final bill for checkout",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        
        print(f"OrderBot initialized with vector store: {self.vector_store_id}")
    
    def search_menu(self, query: str) -> str:
        """Search menu using vector store"""
        try:
            # Create an assistant that can search the vector store
            assistant = client.beta.assistants.create(
                name="Highland Menu Search",
                instructions=f"""You are an Ordering staff at the Highland coffee Shop. Use the vector store to search for accurate menu information for ordering.
                
IMPORTANT: Only provide information that exists in the uploaded menu data. Do not make up sizes, prices, or items.
                
When searching, provide:
- Exact item names as they appear in the menu
- Accurate sizes available (only S, M, L - NOT XL unless specifically mentioned in menu)
- Correct prices in VND 
- Any special notes or descriptions
- Please always response in Paragraph sturture, any don't use any symbol in your response.

Search query: {query}""",
                model="gpt-4o",
                tools=[{"type": "file_search"}],
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [self.vector_store_id]
                    }
                }
            )
            
            # Create a thread
            thread = client.beta.threads.create()
            
            # Add the search query message
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"Search the Highland Coffee menu for: {query}"
            )
            
            # Run the assistant
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Wait for completion
            while run.status in ['queued', 'in_progress']:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=run.thread_id,
                    run_id=run.id
                )
            
            if run.status == 'completed':
                # Get the response
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                response = messages.data[0].content[0].text.value
                
                # Clean up
                client.beta.assistants.delete(assistant.id)
                client.beta.threads.delete(thread.id)
                
                return response
            else:
                # Clean up on error
                client.beta.assistants.delete(assistant.id)
                client.beta.threads.delete(thread.id)
                return f"Search failed with status: {run.status}"
                
        except Exception as e:
            return f"Error searching menu: {e}"
    
    def add_to_order(self, items: List[Dict]) -> str:
        """Add items to current order"""
        added_items = []
        
        for item_data in items:
            try:
                # Check if same item exists and merge
                existing_item = None
                for order_item in self.current_order:
                    if (order_item.name == item_data["name"] and 
                        order_item.size == item_data["size"] and 
                        order_item.modifications == item_data.get("modifications", [])):
                        existing_item = order_item
                        break
                
                if existing_item:
                    existing_item.quantity += item_data["quantity"]
                    added_items.append(f"Updated {item_data['name']} quantity to {existing_item.quantity}")
                else:
                    order_item = OrderItem(
                        name=item_data["name"],
                        size=item_data["size"],
                        quantity=item_data["quantity"],
                        unit_price=item_data["unit_price"],
                        modifications=item_data.get("modifications", [])
                    )
                    self.current_order.append(order_item)
                    added_items.append(f"Added {item_data['name']} ({item_data['size']}) x{item_data['quantity']}")
                    
            except Exception as e:
                added_items.append(f"Error adding {item_data.get('name', 'item')}: {e}")
        
        return "\n".join(added_items)
    
    def show_current_order(self) -> str:
        """Show current order summary"""
        if not self.current_order:
            return "Đơn hàng trống / Order is empty"
        
        order_text = "CURRENT ORDER / ĐƠN HÀNG HIỆN TẠI:\n"
        order_text += "=" * 40 + "\n"
        
        for i, item in enumerate(self.current_order, 1):
            mods = f" ({', '.join(item.modifications)})" if item.modifications else ""
            order_text += f"{i}. {item.name} ({item.size}) x{item.quantity}{mods} - {item.total_price:,}đ\n"
        
        total = sum(item.total_price for item in self.current_order)
        order_text += "=" * 40 + "\n"
        order_text += f"TOTAL / TỔNG: {total:,}đ"
        
        return order_text
    
    def clear_order(self) -> str:
        """Clear the current order"""
        self.current_order.clear()
        return "Đã xóa đơn hàng / Order cleared"
    
    def generate_bill(self) -> str:
        """Generate final bill"""
        if not self.current_order:
            return "No items to bill / Không có món để thanh toán"
        
        bill = "HOÁ ĐƠN THANH TOÁN / PAYMENT BILL\n"
        bill += "Highland Coffee\n"
        bill += "=" * 40 + "\n"
        bill += f"Thời gian / Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        bill += "=" * 40 + "\n"
        
        for i, item in enumerate(self.current_order, 1):
            mods = f" ({', '.join(item.modifications)})" if item.modifications else ""
            bill += f"{i}. {item.name} ({item.size}) x{item.quantity}{mods}\n"
            bill += f"   {item.unit_price:,}đ x {item.quantity} = {item.total_price:,}đ\n"
        
        total = sum(item.total_price for item in self.current_order)
        bill += "=" * 40 + "\n"
        bill += f"TỔNG CỘNG / TOTAL: {total:,}đ\n"
        bill += "=" * 40 + "\n"
        bill += "Cảm ơn quý khách! / Thank you!\n"
        
        return bill
    
    def call_function(self, function_name: str, arguments: Dict) -> str:
        """Execute the requested function"""
        if function_name == "search_menu":
            return self.search_menu(arguments["query"])
        elif function_name == "add_to_order":
            return self.add_to_order(arguments["items"])
        elif function_name == "show_current_order":
            return self.show_current_order()
        elif function_name == "clear_order":
            return self.clear_order()
        elif function_name == "generate_bill":
            return self.generate_bill()
        else:
            return f"Unknown function: {function_name}"
    
    def clean_response(self, text: str) -> str:
        """Clean LLM response by removing unwanted symbols"""
        if not text:
            return text
        
        # Remove unwanted symbols but keep basic punctuation and dashes
        # Remove: * / \ _ ` ~ | { } [ ] < > + = ^ % & # @
        # Keep: - ! ? . , : ; ( ) " ' space and alphanumeric
        cleaned = re.sub(r'[*/\\\_`~|{}[\]<>+=^%&#@]', '', text)
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def process_message(self, user_input: str) -> str:
        """Process user message with LLM and tools"""
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # System message for the assistant
        system_message = {
            "role": "system",
            "content": f"""You are a helpful Highland Coffee ordering assistant. You can speak both Vietnamese and English.

You have access to these tools:
- search_menu: Search Highland Coffee menu for items, prices, sizes
- add_to_order: Add items to customer's order (make sure to get correct prices from menu first)
- show_current_order: Show current order summary
- clear_order: Clear the order
- generate_bill: Generate final bill for checkout

CRITICAL RULES:
- NEVER make up menu information, sizes, or prices
- ALWAYS use search_menu tool FIRST before mentioning any specific items, sizes, or prices
- Only provide information that comes from the vector store search results
- Do not assume sizes like XL exist - only mention sizes that are found in search results
- When customer asks about items, ALWAYS search first, then provide accurate information

RESPONSE FORMATTING:
- Do NOT use these symbols in responses: * / \\ _ ` ~ | {{ }} [ ] < > + = ^ % & # @
- You can use basic punctuation: - ! ? . , : ; ( ) " '
- Keep responses clean and simple

Guidelines:
- Always be friendly and helpful
- When customer asks about items, search the menu first to get accurate information
- When adding items, make sure to get the correct price from menu search
- Ask for clarification on size if not specified (but only offer sizes found in search)
- Support both Vietnamese and English
- For recommendations, search the menu and suggest popular items based on search results
- Handle quantities and modifications naturally

Current vector store ID: {self.vector_store_id}
"""
        }
        
        # Prepare messages for LLM
        messages = [system_message] + self.conversation_history[-10:]  # Keep last 10 exchanges
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )
            
            response_message = response.choices[0].message
            
            # Handle tool calls
            if response_message.tool_calls:
                # Add assistant message with tool calls to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in response_message.tool_calls]
                })
                
                # Execute tool calls
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the function
                    function_result = self.call_function(function_name, function_args)
                    
                    # Add function result to history
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": function_result
                    })
                
                # Get final response after tool execution
                final_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[system_message] + self.conversation_history[-15:],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                final_message = final_response.choices[0].message.content
                cleaned_final = self.clean_response(final_message)
                self.conversation_history.append({"role": "assistant", "content": cleaned_final})
                return cleaned_final
            
            else:
                # No tool calls, just return the response
                cleaned_response = self.clean_response(response_message.content)
                self.conversation_history.append({"role": "assistant", "content": cleaned_response})
                return cleaned_response
                
        except Exception as e:
            error_msg = f"Xin lỗi, có lỗi xảy ra: {e}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def reset_conversation(self):
        """Reset conversation and order"""
        self.conversation_history.clear()
        self.current_order.clear()
    
    def is_order_complete(self) -> bool:
        """Check if order is complete (has bill generated)"""
        return len(self.current_order) > 0 and any(
            "bill" in (msg.get("content") or "").lower() or "hoá đơn" in (msg.get("content") or "").lower()
            for msg in self.conversation_history[-3:]  # Check last 3 messages
        )
    
    def get_order_for_ui(self) -> List[Dict]:
        """Get current order in format suitable for UI"""
        return [item.to_dict() for item in self.current_order]

def main():
    bot = OrderBot()
    
    print("HIGHLAND COFFEE ORDERING SYSTEM")
    print("=" * 50)
    print("Type 'quit' to exit, 'reset' to start new order")
    print("")
    
    # Welcome message
    welcome = bot.process_message("Chào khách hàng mới đến Highland Coffee")
    print("Bot:", welcome)
    print("")
    
    while True:
        try:
            user_input = input("Bạn: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'thoat']:
                print("Hẹn gặp lại!")
                break
            
            if user_input.lower() in ['reset', 'restart']:
                bot.reset_conversation()
                print("Bot: Đã bắt đầu đơn hàng mới!")
                continue
            
            if not user_input:
                continue
            
            response = bot.process_message(user_input)
            print("Bot:", response)
            
            # Check if order seems complete
            if bot.is_order_complete():
                print("\n[Hệ thống: Đơn hàng hoàn thành. Voice module có thể dừng.]")
                bot.reset_conversation()
            
            print("")
            
        except KeyboardInterrupt:
            print("\n\nCảm ơn bạn! Hẹn gặp lại!")
            break
        except Exception as e:
            print(f"Lỗi: {e}")
            print("Vui lòng thử lại!")

if __name__ == "__main__":
    main() 