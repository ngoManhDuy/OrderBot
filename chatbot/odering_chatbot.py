import os
import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
from chatbot.vector_store_manager import VectorStoreManager

# Load environment variables
load_dotenv()
client = OpenAI()

@dataclass
class OrderItem:
    name: str
    size: str
    price: int
    quantity: int = 1

@dataclass
class ConversationContext:
    current_intent: Optional[str] = None
    pending_item: Optional[str] = None
    pending_size: Optional[str] = None
    last_user_message: str = ""
    conversation_history: List[Dict] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []

class ImprovedHighlandBot:
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.vector_store_id = self.vector_manager.get_vector_store_id()
        self.context = ConversationContext()
        self.current_order: List[OrderItem] = []
        
        print(f"🤖 Bot initialized with vector store: {self.vector_store_id}")
    
    def search_menu_simple(self, query: str) -> str:
        """Simple direct search without Assistant API - faster and cleaner."""
        try:
            # Get relevant documents from vector store
            # Using a simple completion with system context instead of assistants
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là nhân viên bán hàng tại Highland Coffee. 
                        Trả lời ngắn gọn, thân thiện bằng tiếng Việt và tiếng anh.
                        Dựa vào thông tin menu để tư vấn chính xác về đồ uống, giá cả, kích cỡ.
                        KHÔNG hiển thị nguồn trích dẫn hay số thứ tự tài liệu.
                        Đôi lúc khách hàng sẽ đưa ra một số thông tin hơi khó hiểu, không nằm trong menu. Trong trường hợp này,
                        bạn hãy lịch sự, hỏi và làm rõ yêu cầu của khách hàng nhé!
                        Lưu ý là hãy CHỈ đưa ra thông tin hữu ích và ngắn gọn nhé. Hãy hạn chế sử dụng các ký hiệu, hãy chỉ viết lời cho tôi"""
                    },
                    {
                        "role": "user", 
                        "content": f"Menu Highland Coffee: {self.get_menu_context()}\n\nKhách hỏi: {query}"
                    }
                ],
                max_tokens=1000,
                temperature=0.2
            )
            
            # Clean response - remove citations
            response_text = response.choices[0].message.content.strip()
            return self.clean_response(response_text)
            
        except Exception as e:
            print(f"Error in search: {e}")
            return "Xin lỗi, tôi không thể tìm thông tin này lúc này. Bạn có thể hỏi về món khác không?"
    
    def get_menu_context(self) -> str:
        """Get basic menu context - you can expand this with cached menu data."""
        return """Highland Coffee Menu:
        
        Cà Phê Pha Phin: Phin Đen, Phin Sữa, Bạc Xíu - Size S,M,L (29k-39k)
        Cà Phê Espresso: Americano, Latte, Cappuccino, Mocha - Size M,L,XL (35k-69k) 
        Đồ Uống Đá Xay: Freeze Sô-cô-la, Cookies & Cream - Size S,M,L (49k-65k)
        Trà: Trà Sen Vàng, Trà Thạch Đào - Size S,M,L (39k-49k)
        Khác: Chanh Đá Xay, Sô-cô-la Đá (39k-54k)"""
    
    def clean_response(self, text: str) -> str:
        """Remove citations and unnecessary formatting."""
        # Remove citation patterns like 【4:0†source】
        text = re.sub(r'【[^】]*】', '', text)
        # Remove source references
        text = re.sub(r'\[source[^\]]*\]', '', text)
        text = re.sub(r'\(\d+:\d+†[^)]*\)', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def detect_intent(self, user_input: str) -> str:
        """Detect user intent based on input and context."""
        user_input_lower = user_input.lower()
        
        # Order management
        if any(word in user_input_lower for word in ["đơn hàng", "order"]):
            return "show_order"
        elif any(word in user_input_lower for word in ["xóa đơn", "clear", "hủy"]):
            return "clear_order"
        
        # Size selection context
        if self.context.pending_item and any(word in user_input_lower for word in ["cỡ", "size", "nhỏ", "vừa", "lớn", "s", "m", "l"]):
            return "selecting_size"
        
        # Ordering intent
        if any(word in user_input_lower for word in ["muốn", "cho tôi", "đặt", "gọi", "order"]):
            return "ordering"
        
        # Information seeking
        if any(word in user_input_lower for word in ["giá", "price", "tiền", "bao nhiêu"]):
            return "asking_price"
        elif any(word in user_input_lower for word in ["gợi ý", "recommend", "tư vấn"]):
            return "asking_recommendation"
        
        return "general_inquiry"
    
    def extract_item_and_size(self, text: str) -> tuple:
        """Extract item name and size from user input."""
        text_lower = text.lower()
        
        # Common items
        items = {
            "freeze sô cô la": "Freeze Sô-cô-la",
            "freeze socola": "Freeze Sô-cô-la", 
            "phin sữa": "Phin Sữa",
            "phin đen": "Phin Đen",
            "latte": "Latte",
            "cappuccino": "Cappuccino",
            "americano": "Americano"
        }
        
        # Size mappings
        sizes = {
            "nhỏ": "S", "s": "S", "small": "S",
            "vừa": "M", "m": "M", "medium": "M", 
            "lớn": "L", "l": "L", "large": "L"
        }
        
        found_item = None
        found_size = None
        
        for key, item in items.items():
            if key in text_lower:
                found_item = item
                break
        
        for key, size in sizes.items():
            if key in text_lower:
                found_size = size
                break
                
        return found_item, found_size
    
    def add_to_order(self, item: str, size: str, price: int):
        """Add item to order."""
        order_item = OrderItem(name=item, size=size, price=price)
        self.current_order.append(order_item)
        return f"✅ Đã thêm {item} cỡ {size} ({price:,}đ) vào đơn hàng!"
    
    def show_order(self) -> str:
        """Show current order."""
        if not self.current_order:
            return "🛒 Đơn hàng của bạn đang trống."
        
        order_text = "🛒 **ĐƠN HÀNG HIỆN TẠI:**\n"
        order_text += "=" * 30 + "\n"
        
        total = 0
        for i, item in enumerate(self.current_order, 1):
            price = item.price * item.quantity
            total += price
            order_text += f"{i}. {item.name} ({item.size}) x{item.quantity} - {price:,}đ\n"
        
        order_text += "=" * 30 + "\n"
        order_text += f"💰 **TỔNG CỘNG: {total:,}đ**"
        
        return order_text
    
    def clear_order(self) -> str:
        """Clear current order."""
        self.current_order = []
        self.context = ConversationContext()
        return "✅ Đã xóa đơn hàng!"
    
    def process_user_input(self, user_input: str) -> str:
        """Main processing function with conversation memory."""
        self.context.last_user_message = user_input
        intent = self.detect_intent(user_input)
        
        # Add to conversation history
        self.context.conversation_history.append({
            "role": "user",
            "content": user_input,
            "intent": intent,
            "timestamp": time.time()
        })
        
        # Process based on intent
        if intent == "show_order":
            response = self.show_order()
            
        elif intent == "clear_order":
            response = self.clear_order()
            
        elif intent == "selecting_size":
            response = self.handle_size_selection(user_input)
            
        elif intent == "ordering":
            response = self.handle_ordering(user_input)
            
        else:
            # General inquiry - search menu
            response = self.search_menu_simple(user_input)
            
            # Check if this creates a new ordering context
            item, _ = self.extract_item_and_size(user_input)
            if item and "muốn" in user_input.lower():
                self.context.current_intent = "ordering"
                self.context.pending_item = item
        
        # Add response to history
        self.context.conversation_history.append({
            "role": "assistant", 
            "content": response,
            "timestamp": time.time()
        })
        
        return response
    
    def handle_size_selection(self, user_input: str) -> str:
        """Handle size selection when user specifies size."""
        if not self.context.pending_item:
            return "Bạn muốn chọn kích cỡ cho món gì ạ?"
        
        _, size = self.extract_item_and_size(user_input)
        
        if size:
            # Get price for the item and size (simplified)
            price = self.get_price(self.context.pending_item, size)
            
            confirm_msg = f"Bạn muốn gọi {self.context.pending_item} cỡ {size} ({price:,}đ) không?"
            self.context.pending_size = size
            
            return confirm_msg
        else:
            return f"Bạn muốn {self.context.pending_item} cỡ nào ạ? (Nhỏ/Vừa/Lớn)"
    
    def handle_ordering(self, user_input: str) -> str:
        """Handle ordering process."""
        item, size = self.extract_item_and_size(user_input)
        
        if item and size:
            price = self.get_price(item, size)
            return self.add_to_order(item, size, price)
        elif item:
            self.context.pending_item = item
            return f"Bạn muốn {item} cỡ nào ạ? (Nhỏ: S, Vừa: M, Lớn: L)"
        else:
            return self.search_menu_simple(user_input)
    
    def get_price(self, item: str, size: str) -> int:
        """Get price for item and size - simplified pricing."""
        # Simplified pricing logic
        base_prices = {
            "Freeze Sô-cô-la": {"S": 49000, "M": 59000, "L": 65000},
            "Phin Sữa": {"S": 29000, "M": 35000, "L": 39000},
            "Latte": {"M": 55000, "L": 65000, "XL": 69000},
            "Cappuccino": {"M": 55000, "L": 65000, "XL": 69000}
        }
        
        if item in base_prices and size in base_prices[item]:
            return base_prices[item][size]
        
        # Default fallback
        return {"S": 39000, "M": 49000, "L": 59000}.get(size, 49000)

def print_welcome():
    """Print welcome message."""
    print("\n" + "="*60)
    print("🌟 CHÀO MỪNG ĐẾN VỚI HIGHLAND COFFEE! 🌟")
    print("="*60)
    print("Tôi là trợ lý AI với trí nhớ, sẵn sàng giúp bạn đặt món!")
    print("\n📋 HƯỚNG DẪN:")
    print("• Hỏi menu: 'Có những đồ uống gì?'")
    print("• Đặt món: 'Tôi muốn freeze sô cô la cỡ nhỏ'") 
    print("• Xem đơn: 'đơn hàng'")
    print("• Xóa đơn: 'xóa đơn'")
    print("• Thoát: 'quit'")
    print("="*60)

def main():
    # Initialize bot
    bot = ImprovedHighlandBot()
    
    print_welcome()
    print("💬 Bạn có thể bắt đầu hỏi ngay!\n")
    
    while True:
        try:
            user_input = input("👤 Bạn: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'thoát', 'bye']:
                print("\n🙏 Cảm ơn bạn đã sử dụng Highland Coffee!")
                print("Hẹn gặp lại! ☕️\n")
                break
            
            if not user_input:
                continue
            
            # Process with improved bot
            response = bot.process_user_input(user_input)
            
            print("🤖 Highland Bot:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            print()
            
        except KeyboardInterrupt:
            print("\n\n🙏 Cảm ơn bạn! Hẹn gặp lại! ☕️\n")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            print("Vui lòng thử lại!\n")

if __name__ == "__main__":
    main() 