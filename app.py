import streamlit as st
import cv2
import requests
import base64
import json
import os
import tempfile
from datetime import datetime
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Shopping & Food Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# DARK MODE CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1d27;
    }

    /* Cards */
    .feature-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #2e3250;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Mode buttons */
    .mode-btn {
        background: linear-gradient(135deg, #6c63ff, #4facfe);
        color: white;
        padding: 15px 30px;
        border-radius: 12px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        width: 100%;
        margin: 5px 0;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1a2035, #1e2847);
        border-left: 4px solid #6c63ff;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        font-size: 16px;
        line-height: 1.8;
    }

    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #0d2818, #1a3a2a);
        border-left: 4px solid #00c853;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }

    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #2a1a00, #3a2800);
        border-left: 4px solid #ff9800;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #6c63ff, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px;
        font-weight: 900;
        text-align: center;
        padding: 20px 0;
    }

    /* Sub header */
    .sub-header {
        color: #8892b0;
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1d27;
        color: #8892b0;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6c63ff, #4facfe);
        color: white;
    }

    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background-color: #1e2130;
        color: white;
        border: 1px solid #2e3250;
        border-radius: 8px;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6c63ff, #4facfe);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        font-size: 15px;
        width: 100%;
        transition: all 0.3s;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(108, 99, 255, 0.4);
    }

    /* Feature badge */
    .badge {
        background: linear-gradient(135deg, #6c63ff, #4facfe);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }

    /* Metric cards */
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2e3250;
    }

    .metric-number {
        font-size: 36px;
        font-weight: 900;
        background: linear-gradient(135deg, #6c63ff, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
DATA_FILE = "shopping_data.json"

def load_data():
    """Load saved shopping data"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {
        "history": [],
        "budget": {"monthly_limit": 0, "spent": 0},
        "wishlist": []
    }

def save_data(data):
    """Save shopping data"""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def image_to_base64(image):
    """Convert PIL image to base64"""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def ask_llava(image, question):
    """Send image + question to LLaVA"""
    try:
        img_b64 = image_to_base64(image)
        payload = {
            "model": "llava",
            "prompt": question,
            "images": [img_b64],
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "Could not get response")
        return f"Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama not running. Please start Ollama first: run 'ollama serve' in terminal"
    except Exception as e:
        return f"Error: {str(e)}"

def ask_llama(prompt):
    """Send text prompt to LLaMA 3"""
    try:
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "Could not get response")
        return f"Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "⚠️ Ollama not running. Please start Ollama first: run 'ollama serve' in terminal"
    except Exception as e:
        return f"Error: {str(e)}"

def scan_barcode(image):
    """Scan barcode from image using pyzbar"""
    try:
        from pyzbar import pyzbar
        img_array = np.array(image)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        barcodes = pyzbar.decode(img_gray)
        if barcodes:
            results = []
            for barcode in barcodes:
                data = barcode.data.decode("utf-8")
                barcode_type = barcode.type
                results.append(f"Type: {barcode_type} | Data: {data}")
            return "\n".join(results)
        return "No barcode detected in image"
    except ImportError:
        return "pyzbar not installed. Run: pip install pyzbar"
    except Exception as e:
        return f"Error scanning barcode: {str(e)}"

def speak_text(text):
    """Convert text to speech using gTTS"""
    try:
        from gtts import gTTS
        import pygame
        tts = gTTS(text=text[:500], lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts.save(f.name)
            return f.name
    except ImportError:
        return None
    except Exception:
        return None

def transcribe_voice(audio_file):
    """Transcribe voice using Whisper"""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        return result["text"]
    except ImportError:
        return "Whisper not installed. Run: pip install openai-whisper"
    except Exception as e:
        return f"Error: {str(e)}"

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_product_context" not in st.session_state:
    st.session_state.current_product_context = ""
if "current_food_context" not in st.session_state:
    st.session_state.current_food_context = ""
if "data" not in st.session_state:
    st.session_state.data = load_data()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🛒 AI Smart Shopping & Food Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your Complete Food Intelligence Companion — Supermarket + Street Food</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    st.markdown("---")

    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{len(st.session_state.data['history'])}</div>
            <div style="color:#8892b0;font-size:12px">Scans Done</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{len(st.session_state.data['wishlist'])}</div>
            <div style="color:#8892b0;font-size:12px">Wishlist</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💰 Monthly Budget")
    budget_limit = st.number_input(
        "Set Budget (₹)",
        min_value=0,
        value=int(st.session_state.data["budget"]["monthly_limit"]),
        step=500
    )
    if budget_limit != st.session_state.data["budget"]["monthly_limit"]:
        st.session_state.data["budget"]["monthly_limit"] = budget_limit
        save_data(st.session_state.data)

    spent = st.session_state.data["budget"]["spent"]
    if budget_limit > 0:
        progress = min(spent / budget_limit, 1.0)
        st.progress(progress)
        remaining = budget_limit - spent
        color = "🟢" if remaining > budget_limit * 0.3 else "🔴"
        st.markdown(f"{color} Spent: ₹{spent} / ₹{budget_limit}")
        st.markdown(f"Remaining: ₹{remaining}")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    voice_enabled = st.toggle("🔊 Voice Output", value=True)
    st.markdown("---")
    st.markdown("### 🏷️ Features")
    features = ["📷 Label Scanner", "📦 Barcode Reader", "✍️ List Scanner",
                "🎤 Voice Search", "🔊 Voice Verdict", "🤝 Hands-Free",
                "💡 Worth Buying?", "❤️ Health Score", "⚖️ Compare",
                "🌿 Allergens", "📅 Expiry Check", "💰 Price/Unit",
                "🍜 Dish ID", "🛡️ Food Safety", "💵 Price Guide",
                "🌾 Allergen Q&A", "🍷 Food Pairing", "📖 Dish Story"]
    for f in features:
        st.markdown(f'<span class="badge">{f}</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛒 Shopping Mode",
    "🍜 Street Food Mode",
    "🎤 Voice Assistant",
    "📊 My Tracker",
    "📋 Shopping History"
])

# ═══════════════════════════════════════════════
# TAB 1 — SHOPPING MODE
# ═══════════════════════════════════════════════
with tab1:
    st.markdown("## 🛒 Shopping Assistant")
    st.markdown("Point your camera at any product — ask anything!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📷 Scan Product")
        scan_method = st.radio(
            "Input Method",
            ["📸 Camera", "🖼️ Upload Image"],
            horizontal=True
        )

        image = None
        if scan_method == "📸 Camera":
            camera_image = st.camera_input("Point at product")
            if camera_image:
                image = Image.open(camera_image)
        else:
            uploaded = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])
            if uploaded:
                image = Image.open(uploaded)

        if image:
            st.image(image, caption="Scanned Product", use_container_width=True)

            # Barcode scanning
            st.markdown("#### 📦 Barcode Scanner")
            if st.button("🔍 Scan Barcode"):
                with st.spinner("Scanning barcode..."):
                    barcode_result = scan_barcode(image)
                    st.markdown(f"""
                    <div class="result-box">
                    <b>📦 Barcode Result:</b><br>{barcode_result}
                    </div>
                    """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🤖 Ask About This Product")

        if image:
            # Quick action buttons
            st.markdown("#### ⚡ Quick Actions")
            quick_col1, quick_col2 = st.columns(2)

            with quick_col1:
                if st.button("💡 Worth Buying?"):
                    with st.spinner("Analysing..."):
                        vision = ask_llava(image, "Describe this product in detail including name, brand, price if visible, ingredients, and any other details you can see on the label.")
                        st.session_state.current_product_context = vision
                        prompt = f"""Based on this product: {vision}

Tell me:
1. What is this product?
2. Is it worth buying? (value for money)
3. Pros and cons
4. Overall verdict (Buy / Skip / Maybe)

Be honest and concise."""
                        result = ask_llama(prompt)
                        st.session_state.chat_history.append({"role": "assistant", "content": result, "type": "shopping"})
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

                if st.button("❤️ Health Score"):
                    with st.spinner("Checking health score..."):
                        vision = ask_llava(image, "Read all ingredients, nutritional information, sugar content, sodium, fats from this product label.")
                        prompt = f"""Based on this product: {vision}

Give me:
1. Health score out of 10
2. Main health concerns
3. Who should avoid this
4. Healthier alternatives

Be direct and honest."""
                        result = ask_llama(prompt)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

                if st.button("📅 Expiry Check"):
                    with st.spinner("Checking expiry..."):
                        vision = ask_llava(image, "Find and read the expiry date, best before date, or manufacturing date on this product.")
                        prompt = f"""Based on: {vision}

Tell me:
1. Expiry/best before date
2. Is it safe to buy/consume now?
3. How long until expiry
4. Storage advice

Today's date: {datetime.now().strftime('%d %B %Y')}"""
                        result = ask_llama(prompt)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

            with quick_col2:
                if st.button("⚖️ Compare Products"):
                    st.info("Upload a second product image below to compare")

                if st.button("🌿 Allergens"):
                    with st.spinner("Checking allergens..."):
                        vision = ask_llava(image, "Read all ingredients and allergen warnings from this product label.")
                        prompt = f"""Based on: {vision}

List:
1. All allergens present
2. May contain warnings
3. Safe for: vegetarians/vegans/gluten-free/diabetics
4. Hidden allergens to watch out for"""
                        result = ask_llama(prompt)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

                if st.button("💰 Price Per Unit"):
                    with st.spinner("Calculating..."):
                        vision = ask_llava(image, "Read the price, weight/volume/quantity from this product.")
                        prompt = f"""Based on: {vision}

Calculate:
1. Price per gram/ml/unit
2. Is this good value compared to typical market prices?
3. Better value size/brand recommendations"""
                        result = ask_llama(prompt)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

                if st.button("🔄 Alternatives"):
                    with st.spinner("Finding alternatives..."):
                        vision = ask_llava(image, "What product is this? Brand, type, price if visible.")
                        prompt = f"""Based on: {vision}

Suggest:
1. 3 cheaper alternatives
2. 3 healthier alternatives  
3. Best overall alternative and why"""
                        result = ask_llama(prompt)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

            # Free Q&A
            st.markdown("---")
            st.markdown("#### 💬 Ask Anything About This Product")
            user_question = st.text_input("Type your question...", placeholder="Is this safe for diabetics?")
            if st.button("🚀 Ask") and user_question:
                with st.spinner("Thinking..."):
                    if st.session_state.current_product_context:
                        prompt = f"""Product context: {st.session_state.current_product_context}

User question: {user_question}

Answer helpfully and honestly."""
                    else:
                        vision = ask_llava(image, "Describe this product completely.")
                        st.session_state.current_product_context = vision
                        prompt = f"""Product: {vision}

User question: {user_question}

Answer helpfully and honestly."""
                    result = ask_llama(prompt)
                    st.markdown(f'<div class="result-box"><b>Q: {user_question}</b><br><br>{result}</div>', unsafe_allow_html=True)

                    # Save to history
                    st.session_state.data["history"].append({
                        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "type": "Shopping",
                        "question": user_question,
                        "answer": result[:200] + "..."
                    })
                    save_data(st.session_state.data)

            # Add to wishlist
            st.markdown("---")
            wishlist_name = st.text_input("Product name for wishlist")
            if st.button("❤️ Add to Wishlist") and wishlist_name:
                st.session_state.data["wishlist"].append({
                    "name": wishlist_name,
                    "added": datetime.now().strftime("%d/%m/%Y")
                })
                save_data(st.session_state.data)
                st.success(f"✅ {wishlist_name} added to wishlist!")

        else:
            st.markdown("""
            <div class="feature-card">
            <h3>👆 How to use Shopping Mode</h3>
            <ol>
                <li>Point camera at any product OR upload an image</li>
                <li>Click any Quick Action button</li>
                <li>Or type any question in the chat box</li>
                <li>Get instant AI-powered answers!</li>
            </ol>
            <br>
            <b>Try asking:</b><br>
            • "What is this product?"<br>
            • "Is this healthy for my child?"<br>
            • "Is this worth ₹150?"<br>
            • "Does this have gluten?"<br>
            • "How long will this last?"
            </div>
            """, unsafe_allow_html=True)

    # Handwritten shopping list scanner
    st.markdown("---")
    st.markdown("### ✍️ Handwritten Shopping List Scanner")
    list_col1, list_col2 = st.columns([1, 1])
    with list_col1:
        list_image = st.file_uploader("Upload your handwritten shopping list", type=["jpg", "jpeg", "png"], key="list_upload")
        if list_image:
            list_img = Image.open(list_image)
            st.image(list_img, caption="Your Shopping List", use_container_width=True)

    with list_col2:
        if list_image and st.button("📋 Read My List"):
            with st.spinner("Reading your handwritten list..."):
                result = ask_llava(list_img, "Read this handwritten shopping list carefully. List every item you can see written on it.")
                prompt = f"""Shopping list items: {result}

For each item:
1. Confirm the item name
2. Suggest what to look for when buying
3. Estimated price range in India (₹)
4. Any buying tips

Format clearly."""
                final = ask_llama(prompt)
                st.markdown(f'<div class="result-box">{final}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 2 — STREET FOOD MODE
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("## 🍜 Street Food Judge")
    st.markdown("Point your camera at any street food — discover, understand, enjoy!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📷 Capture Street Food")
        food_method = st.radio(
            "Input Method",
            ["📸 Camera", "🖼️ Upload Image"],
            horizontal=True,
            key="food_radio"
        )

        food_image = None
        if food_method == "📸 Camera":
            food_camera = st.camera_input("Point at street food", key="food_camera")
            if food_camera:
                food_image = Image.open(food_camera)
        else:
            food_uploaded = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"], key="food_upload")
            if food_uploaded:
                food_image = Image.open(food_uploaded)

        if food_image:
            st.image(food_image, caption="Street Food", use_container_width=True)

    with col2:
        st.markdown("### 🤖 Food Intelligence")

        if food_image:
            # First scan the food
            food_quick1, food_quick2 = st.columns(2)

            with food_quick1:
                if st.button("🍽️ What Is This?"):
                    with st.spinner("Identifying dish..."):
                        vision = ask_llava(food_image, "What street food dish is this? Describe it in detail including appearance, ingredients visible, cooking method, and any other details.")
                        st.session_state.current_food_context = vision
                        prompt = f"""Based on this street food: {vision}

Tell me:
1. Name of the dish
2. Region/state it belongs to
3. Main ingredients
4. How it's typically made
5. Best time to eat it

Be enthusiastic and informative!"""
                        result = ask_llama(prompt)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

                if st.button("📖 Dish Story"):
                    with st.spinner("Finding the story..."):
                        if not st.session_state.current_food_context:
                            st.session_state.current_food_context = ask_llava(food_image, "What dish is this?")
                        prompt = f"""For this street food: {st.session_state.current_food_context}

Tell me a fascinating story about:
1. Historical origin (when and where it started)
2. Cultural significance
3. How it evolved over time
4. Interesting facts most people don't know
5. Famous places to eat this

Write like an engaging food documentary narrator!"""
                        result = ask_llama(prompt)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

                if st.button("🛡️ Safety Tips"):
                    with st.spinner("Checking safety..."):
                        vision = ask_llava(food_image, "Describe the food stall or food preparation visible. Is the food covered? How does it look? What's the cooking environment like?")
                        prompt = f"""Street food stall observation: {vision}

Give me:
1. General food safety assessment based on what's visible
2. 3 things that look good (if any)
3. 3 things to be cautious about (if any)
4. Should I eat here? Overall recommendation
5. Tips for eating street food safely

Be honest but fair."""
                        result = ask_llama(prompt)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

            with food_quick2:
                if st.button("🌾 Allergens Q&A"):
                    with st.spinner("Checking allergens..."):
                        if not st.session_state.current_food_context:
                            st.session_state.current_food_context = ask_llava(food_image, "What dish is this?")
                        prompt = f"""For: {st.session_state.current_food_context}

Tell me about allergens:
1. Common allergens in this dish
2. Is it vegetarian/vegan?
3. Does it contain gluten?
4. Does it contain dairy?
5. Does it contain nuts?
6. What to ask the vendor to confirm

Important: These are based on traditional recipes — always confirm with the seller!"""
                        result = ask_llama(prompt)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

                if st.button("🍷 Best Pairings"):
                    with st.spinner("Finding pairings..."):
                        if not st.session_state.current_food_context:
                            st.session_state.current_food_context = ask_llava(food_image, "What dish is this?")
                        prompt = f"""For: {st.session_state.current_food_context}

Suggest the perfect pairings:
1. Best drink to have with this
2. Best side dish or accompaniment
3. What to eat before or after
4. What NOT to eat with this
5. Perfect time of day to enjoy this

Make it sound delicious!"""
                        result = ask_llama(prompt)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

                if st.button("💵 Fair Price?"):
                    with st.spinner("Checking price..."):
                        if not st.session_state.current_food_context:
                            st.session_state.current_food_context = ask_llava(food_image, "What dish is this?")
                        price_paid = st.session_state.get("price_paid", 0)
                        prompt = f"""For this street food: {st.session_state.current_food_context}

Tell me:
1. Typical price range in Tamil Nadu/South India (₹)
2. What affects the price (location, quality, portion)
3. Is ₹{price_paid} a fair price? (if 0, just give typical range)
4. Tips for getting best value

Base answer on real Indian street food prices."""
                        result = ask_llama(prompt)
                        st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

            # Price input
            st.session_state.price_paid = st.number_input("I paid (₹)", min_value=0, step=5, key="price_input")

            # Free Q&A for food
            st.markdown("---")
            st.markdown("#### 💬 Ask Anything About This Food")
            food_question = st.text_input("Type your question...", placeholder="Is this spicy? Can my child eat this?", key="food_q")
            if st.button("🚀 Ask Chef AI") and food_question:
                with st.spinner("Asking the AI chef..."):
                    if not st.session_state.current_food_context:
                        st.session_state.current_food_context = ask_llava(food_image, "Describe this food completely.")
                    prompt = f"""Street food context: {st.session_state.current_food_context}

Question: {food_question}

Answer like a knowledgeable local food expert. Be helpful and specific."""
                    result = ask_llama(prompt)
                    st.markdown(f'<div class="result-box"><b>Q: {food_question}</b><br><br>{result}</div>', unsafe_allow_html=True)

                    # Save to history
                    st.session_state.data["history"].append({
                        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "type": "Street Food",
                        "question": food_question,
                        "answer": result[:200] + "..."
                    })
                    save_data(st.session_state.data)

        else:
            st.markdown("""
            <div class="feature-card">
            <h3>👆 How to use Street Food Mode</h3>
            <ol>
                <li>Point camera at any street food dish or stall</li>
                <li>Click any Quick Action button</li>
                <li>Or ask any food question in the chat</li>
                <li>Discover the food, its story, safety, and more!</li>
            </ol>
            <br>
            <b>Try asking:</b><br>
            • "What is this dish called?"<br>
            • "Is this safe to eat?"<br>
            • "Tell me the history of this food"<br>
            • "Does this have gluten?"<br>
            • "What drink goes with this?"
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 3 — VOICE ASSISTANT
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("## 🎤 Voice Assistant")
    st.markdown("Ask anything by voice — get answers spoken back to you!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🎙️ Voice Input")
        st.markdown("""
        <div class="feature-card">
        <h4>How Voice Works:</h4>
        <ol>
            <li>Record your question using the audio recorder</li>
            <li>AI transcribes your speech</li>
            <li>AI answers your question</li>
            <li>Answer is spoken back to you</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        audio_file = st.file_uploader("Upload audio question (WAV/MP3)", type=["wav", "mp3", "m4a"])

        if audio_file and st.button("🎤 Transcribe & Answer"):
            with st.spinner("Processing your voice..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_file.read())
                    temp_path = f.name

                transcribed = transcribe_voice(temp_path)
                st.markdown(f"""
                <div class="success-box">
                <b>🎤 You said:</b><br>{transcribed}
                </div>
                """, unsafe_allow_html=True)

                # Answer the question
                answer = ask_llama(f"Answer this shopping or food question helpfully: {transcribed}")
                st.markdown(f'<div class="result-box"><b>🤖 AI Answer:</b><br>{answer}</div>', unsafe_allow_html=True)

                # Speak the answer
                if voice_enabled:
                    audio_path = speak_text(answer)
                    if audio_path:
                        st.audio(audio_path)

    with col2:
        st.markdown("### 💬 Hands-Free Text Mode")
        st.markdown("Type your question — AI answers and speaks back")

        hands_free_q = st.text_area("Your question", placeholder="Is Maggi healthy? What is dosa made of? Should I buy this product?", height=100)

        if st.button("🚀 Get Answer + Speak"):
            if hands_free_q:
                with st.spinner("Thinking and speaking..."):
                    answer = ask_llama(f"""Answer this shopping or food question in a helpful, conversational way:

Question: {hands_free_q}

Give a clear, practical answer.""")
                    st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)

                    if voice_enabled:
                        audio_path = speak_text(answer)
                        if audio_path:
                            st.audio(audio_path)
                            st.success("🔊 Answer spoken! Press play to hear it")

        st.markdown("---")
        st.markdown("### 💡 Try These Questions")
        sample_questions = [
            "What should I look for when buying olive oil?",
            "Is biryani healthy to eat daily?",
            "How do I know if an egg is fresh?",
            "What is the best time to buy vegetables?",
            "Is Maggi safe for children?",
            "What does dosa taste like?"
        ]
        for q in sample_questions:
            if st.button(f"💬 {q}", key=f"sample_{q}"):
                with st.spinner("Answering..."):
                    answer = ask_llama(f"Answer this helpfully: {q}")
                    st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)
                    if voice_enabled:
                        audio_path = speak_text(answer)
                        if audio_path:
                            st.audio(audio_path)

# ═══════════════════════════════════════════════
# TAB 4 — TRACKER
# ═══════════════════════════════════════════════
with tab4:
    st.markdown("## 📊 My Shopping Tracker")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 💰 Budget Tracker")
        add_expense = st.number_input("Add expense (₹)", min_value=0, step=10)
        expense_note = st.text_input("Note (e.g. Groceries, Vegetables)")
        if st.button("➕ Add Expense"):
            if add_expense > 0:
                st.session_state.data["budget"]["spent"] += add_expense
                st.session_state.data["history"].append({
                    "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "type": "Expense",
                    "question": expense_note or "Expense added",
                    "answer": f"₹{add_expense} spent"
                })
                save_data(st.session_state.data)
                limit = st.session_state.data["budget"]["monthly_limit"]
                spent = st.session_state.data["budget"]["spent"]
                if limit > 0 and spent > limit * 0.9:
                    st.warning(f"⚠️ Alert! You've spent ₹{spent} out of ₹{limit} budget!")
                else:
                    st.success(f"✅ ₹{add_expense} added. Total spent: ₹{st.session_state.data['budget']['spent']}")

        if st.button("🔄 Reset Monthly Budget"):
            st.session_state.data["budget"]["spent"] = 0
            save_data(st.session_state.data)
            st.success("Budget reset for new month!")

    with col2:
        st.markdown("### ❤️ My Wishlist")
        if st.session_state.data["wishlist"]:
            for i, item in enumerate(st.session_state.data["wishlist"]):
                wish_col1, wish_col2 = st.columns([3, 1])
                with wish_col1:
                    st.markdown(f"• **{item['name']}** — Added: {item['added']}")
                with wish_col2:
                    if st.button("✅ Got it", key=f"wish_{i}"):
                        st.session_state.data["wishlist"].pop(i)
                        save_data(st.session_state.data)
                        st.rerun()
        else:
            st.markdown("""
            <div class="feature-card">
            <p style="color:#8892b0">No items in wishlist yet.<br>
            Add products from Shopping Mode!</p>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 5 — HISTORY
# ═══════════════════════════════════════════════
with tab5:
    st.markdown("## 📋 Shopping & Food History")

    if st.session_state.data["history"]:
        if st.button("🗑️ Clear History"):
            st.session_state.data["history"] = []
            save_data(st.session_state.data)
            st.rerun()

        for item in reversed(st.session_state.data["history"][-20:]):
            emoji = "🛒" if item["type"] == "Shopping" else "🍜" if item["type"] == "Street Food" else "💰"
            st.markdown(f"""
            <div class="feature-card">
            <small style="color:#8892b0">{emoji} {item['type']} — {item['timestamp']}</small><br>
            <b>{item['question']}</b><br>
            <small style="color:#a0aec0">{item['answer']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="feature-card">
        <p style="color:#8892b0;text-align:center">
        No history yet.<br>
        Start scanning products and food to build your history!
        </p>
        </div>
        """, unsafe_allow_html=True)