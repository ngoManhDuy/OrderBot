import os
import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
from vector_store_manager import VectorStoreManager

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
    preferred_language: str = "auto"  # "auto", "vi", "en"
    
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
            # Get system prompt based on preferred language
            system_prompt = self.get_system_prompt()
            
            # Get relevant documents from vector store
            # Using a simple completion with system context instead of assistants
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": f"Menu Highland Coffee: {self.get_menu_context()}\n\nCustomer asks: {query}"
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
            if self.context.preferred_language == "en":
                return "Sorry, I cannot find this information right now. Can you ask about something else?"
            else:
                return "Xin lỗi, tôi không thể tìm thông tin này lúc này. Bạn có thể hỏi về món khác không?"
    
    def get_menu_context(self) -> str:
        """Get basic menu context - you can expand this with cached menu data."""
        return """Highland Coffee Menu:
        
        Cà Phê Pha Phin: Phin Đen, Phin Sữa, Bạc Xíu - Size S,M,L (29k-39k)
        Cà Phê Espresso: Americano, Latte, Cappuccino, Mocha - Size M,L,XL (35k-69k) 
        Đồ Uống Đá Xay: Freeze Sô-cô-la, Cookies & Cream - Size S,M,L (49k-65k)
        Trà: Trà Sen Vàng, Trà Thạch Đào - Size S,M,L (39k-49k)
        Khác: Chanh Đá Xay, Sô-cô-la Đá (39k-54k)"""
    
    def get_system_prompt(self) -> str:
        """Get system prompt based on user's preferred language."""
        if self.context.preferred_language == "en":
            return """You are a sales staff member at Highland Coffee.
            Respond concisely and friendly in ENGLISH only.
            Based on the menu information, provide accurate advice about drinks, prices, and sizes.
            DO NOT display source citations or document numbers.
            Sometimes customers may provide confusing information not on the menu. In this case,
            please be polite, ask questions and clarify the customer's request!
            Note that you should ONLY provide useful and concise information. Limit the use of symbols, just write words for me."""
        
        elif self.context.preferred_language == "vi":
            return """Bạn là nhân viên bán hàng tại Highland Coffee. 
            Trả lời ngắn gọn, thân thiện bằng tiếng Việt.
            Dựa vào thông tin menu để tư vấn chính xác về đồ uống, giá cả, kích cỡ.
            KHÔNG hiển thị nguồn trích dẫn hay số thứ tự tài liệu.
            Đôi lúc khách hàng sẽ đưa ra một số thông tin hơi khó hiểu, không nằm trong menu. Trong trường hợp này,
            bạn hãy lịch sự, hỏi và làm rõ yêu cầu của khách hàng nhé!
            Lưu ý là hãy CHỈ đưa ra thông tin hữu ích và ngắn gọn nhé. Hãy hạn chế sử dụng các ký hiệu, hãy chỉ viết lời cho tôi."""
        
        else:  # auto mode - detect from context
            return """You are a sales staff member at Highland Coffee.
            Respond concisely and friendly. Match the customer's language preference:
            - If they write in English, respond in English
            - If they write in Vietnamese, respond in Vietnamese  
            - If unclear, default to Vietnamese
            Based on the menu information, provide accurate advice about drinks, prices, and sizes.
            DO NOT display source citations or document numbers."""
    
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
    
    def detect_language(self, user_input: str) -> str:
        """Detect user's preferred language from their input."""
        user_input_lower = user_input.lower()
        
        # Explicit language requests
        if any(phrase in user_input_lower for phrase in [
            "speak english", "use english", "in english", "english please",
            "can you speak english", "switch to english"
        ]):
            return "en"
        
        if any(phrase in user_input_lower for phrase in [
            "nói tiếng việt", "tiếng việt", "vietnamese", "việt nam"
        ]):
            return "vi"
        
        # Language detection based on content
        # Vietnamese indicators
        vietnamese_words = [
            'tôi', 'bạn', 'là', 'và', 'có', 'không', 'được', 'này', 'đó', 'của',
            'cho', 'với', 'về', 'một', 'những', 'từ', 'ở', 'như', 'sẽ', 'nào',
            'muốn', 'cần', 'thích', 'gì', 'ạ', 'em', 'anh', 'chị', 'xin'
        ]
        
        # English indicators  
        english_words = [
            'i', 'you', 'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 
            'was', 'were', 'have', 'has', 'do', 'does', 'will', 'would',
            'can', 'could', 'should', 'want', 'need', 'like', 'what', 'how'
        ]
        
        vietnamese_count = sum(1 for word in vietnamese_words if word in user_input_lower)
        english_count = sum(1 for word in english_words if word in user_input_lower)
        
        if vietnamese_count > english_count:
            return "vi"
        elif english_count > vietnamese_count:
            return "en"
        
        return "auto"  # Cannot determine
    
    def detect_intent(self, user_input: str) -> str:
        """Detect user intent based on input and context."""
        user_input_lower = user_input.lower()
        
        # Order management
        if any(word in user_input_lower for word in ["đơn hàng", "order", "my order", "cart"]):
            return "show_order"
        elif any(word in user_input_lower for word in ["xóa đơn", "clear", "hủy", "cancel", "remove"]):
            return "clear_order"
        
        # Size selection context
        if self.context.pending_item and any(word in user_input_lower for word in [
            "cỡ", "size", "nhỏ", "vừa", "lớn", "s", "m", "l",
            "small", "medium", "large"
        ]):
            return "selecting_size"
        
        # Ordering intent
        if any(word in user_input_lower for word in [
            "muốn", "cho tôi", "đặt", "gọi", "order", "want", "get", "have",
            "i want", "i would like", "can i get", "can i have", "give me"
        ]):
            return "ordering"
        
        # Information seeking
        if any(word in user_input_lower for word in [
            "giá", "price", "tiền", "bao nhiêu", "how much", "cost", "pricing"
        ]):
            return "asking_price"
        elif any(word in user_input_lower for word in [
            "gợi ý", "recommend", "tư vấn", "suggestion", "what do you recommend",
            "what's good", "what should i"
        ]):
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
        
        if self.context.preferred_language == "en":
            return f"✅ Added {item} size {size} ({price:,}đ) to your order!"
        else:
            return f"✅ Đã thêm {item} cỡ {size} ({price:,}đ) vào đơn hàng!"
    
    def show_order(self) -> str:
        """Show current order."""
        if not self.current_order:
            if self.context.preferred_language == "en":
                return "🛒 Your order is empty."
            else:
                return "🛒 Đơn hàng của bạn đang trống."
        
        if self.context.preferred_language == "en":
            order_text = "🛒 **CURRENT ORDER:**\n"
            order_text += "=" * 30 + "\n"
            
            total = 0
            for i, item in enumerate(self.current_order, 1):
                price = item.price * item.quantity
                total += price
                order_text += f"{i}. {item.name} ({item.size}) x{item.quantity} - {price:,}đ\n"
            
            order_text += "=" * 30 + "\n"
            order_text += f"💰 **TOTAL: {total:,}đ**"
        else:
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
        
        if self.context.preferred_language == "en":
            return "✅ Order cleared!"
        else:
            return "✅ Đã xóa đơn hàng!"
    
    def process_user_input(self, user_input: str) -> str:
        """Main processing function with conversation memory."""
        self.context.last_user_message = user_input
        
        # Detect and update language preference
        detected_language = self.detect_language(user_input)
        if detected_language != "auto":
            self.context.preferred_language = detected_language
        
        intent = self.detect_intent(user_input)
        
        # Add to conversation history
        self.context.conversation_history.append({
            "role": "user",
            "content": user_input,
            "intent": intent,
            "language": detected_language,
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
            if self.context.preferred_language == "en":
                return "What item would you like to choose a size for?"
            else:
                return "Bạn muốn chọn kích cỡ cho món gì ạ?"
        
        _, size = self.extract_item_and_size(user_input)
        
        if size:
            # Get price for the item and size (simplified)
            price = self.get_price(self.context.pending_item, size)
            
            if self.context.preferred_language == "en":
                confirm_msg = f"Would you like to order {self.context.pending_item} size {size} ({price:,}đ)?"
            else:
                confirm_msg = f"Bạn muốn gọi {self.context.pending_item} cỡ {size} ({price:,}đ) không?"
            
            self.context.pending_size = size
            return confirm_msg
        else:
            if self.context.preferred_language == "en":
                return f"What size would you like for {self.context.pending_item}? (Small/Medium/Large)"
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
            if self.context.preferred_language == "en":
                return f"What size would you like for {item}? (Small: S, Medium: M, Large: L)"
            else:
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
    
    def get_welcome_message(self) -> str:
        """Get welcome message in appropriate language."""
        if self.context.preferred_language == "en":
            return "Hello! Welcome to Highland Coffee! How can I help you today? ☕️"
        else:
            return "Chào bạn! Chào mừng đến Highland Coffee! Bạn cần tư vấn gì hôm nay? ☕️"

def print_welcome():
    """Print welcome message."""
    print("HIGHLAND COFFEE ORDERING")
    print("="*60)

def main():
    # Initialize bot
    bot = ImprovedHighlandBot()
    
    print_welcome()
    print("💬 Bạn có thể bắt đầu hỏi ngay! / You can start asking now!\n")
    
    # Show initial greeting
    initial_greeting = bot.get_welcome_message()
    print("🤖 Highland Bot:")
    print("-" * 40)
    print(initial_greeting)
    print("-" * 40)
    print()
    
    while True:
        try:
            user_input = input("👤 Bạn: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'thoát', 'bye']:
                print("Hẹn gặp lại! \n")
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