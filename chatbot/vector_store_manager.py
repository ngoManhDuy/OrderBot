import json
import os
import base64
import datetime
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI()

class VectorStoreManager:
    def __init__(self, config_file="vector_store_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load vector store configuration from file."""
        if Path(self.config_file).exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_config(self, vector_store_id, file_id=None):
        """Save vector store configuration to file."""
        config = {
            "vector_store_id": vector_store_id,
            "file_id": file_id,
            "created_at": datetime.datetime.now().isoformat(),
            "menu_file": "menu.jpg"
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.config = config
        print(f"✅ Vector store config saved: {vector_store_id}")
    
    def get_vector_store_id(self):
        """Get existing vector store ID or create new one."""
        if "vector_store_id" in self.config:
            vector_store_id = self.config["vector_store_id"]
            
            # Verify the vector store still exists
            try:
                store = client.vector_stores.retrieve(vector_store_id)
                print(f"✅ Using existing vector store: {vector_store_id}")
                return vector_store_id
            except Exception as e:
                print(f"⚠️ Existing vector store not found, creating new one...")
                # Remove invalid config and create new
                self.config = {}
        
        # Create new vector store
        return self.create_vector_store()
    
    def create_vector_store(self):
        """Create new vector store with menu data."""
        print("🔄 Creating new vector store...")
        
        # Create vector store
        vector_store = client.vector_stores.create(
            name="highlands_menu_analysis",
            metadata={
                "purpose": "Highland Coffee menu information for ordering assistance",
                "author": "Ngo Manh Duy",
            }
        )
        
        vector_store_id = vector_store.id
        print(f"✅ Vector store created: {vector_store_id}")
        
        # Upload menu data
        file_id = self.upload_menu_data(vector_store_id)
        
        # Save configuration
        self.save_config(vector_store_id, file_id)
        
        return vector_store_id
    
    def encode_image(self, image_path: str) -> str:
        """Encode image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def analyze_image(self, image_path: str) -> str:
        """Analyze the menu image and return text description."""
        print(f"🔍 Analyzing image: {image_path}")
        
        base64_image = self.encode_image(image_path)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hãy mô tả chi tiết và đầy đủ về menu Highland Coffee này. "
                        "Bao gồm: tên các món, kích cỡ, giá cả chính xác. "
                        "Hãy tổ chức thông tin rõ ràng theo từng danh mục vàcung cấp thông tin đầy đủ nhất có thể. "
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }],
            max_tokens=1000,
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
    
    def upload_menu_data(self, vector_store_id):
        """Upload menu analysis as text to vector store."""
        menu_file = "menu.png"
        text_filename = "menu_analysis_complete.txt"
        
        if not Path(menu_file).exists():
            raise FileNotFoundError(f"Menu file {menu_file} not found!")
        
        # Analyze the image
        menu_analysis = self.analyze_image(menu_file)
        
        # Create comprehensive text file
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write("# HIGHLAND COFFEE MENU - THÔNG TIN CHI TIẾT\n\n")
            f.write("## Phân tích menu từ hình ảnh:\n\n")
            f.write(menu_analysis)
            f.write("\n\n## Thông tin bổ sung:\n")
            f.write("- Menu này cung cấp thông tin về đồ uống tại Highland Coffee\n")
            f.write("- Bao gồm các loại: Cà phê, Trà, Đồ uống đá xay, Trà , bánh, ...\n")
            f.write("- Giá cả và kích cỡ được cập nhật theo menu mới nhất\n")
            f.write("- Phù hợp cho việc tư vấn và đặt món cho khách hàng\n")
        
        # Upload to vector store
        with open(text_filename, "rb") as f:
            file_stream = BytesIO(f.read())
        file_stream.name = text_filename
        
        file = client.vector_stores.files.upload(
            vector_store_id=vector_store_id,
            file=file_stream
        )
        
        print(f"✅ Menu data uploaded with file ID: {file.id}")
        
        # Clean up temporary file
        os.remove(text_filename)
        
        return file.id

def main():
    """Test the vector store manager."""
    manager = VectorStoreManager()
    vector_store_id = manager.get_vector_store_id()
    print(f"\n🎉 Ready to use vector store: {vector_store_id}")

if __name__ == "__main__":
    main() 