# AI-Enhanced Heart Disease Chatbot Setup Instructions (Google Gemini)

This document provides comprehensive instructions for setting up and using the AI-enhanced cardiovascular consultation system powered by Google Gemini API.

## 🌟 Why Google Gemini?

**Excellent choice!** Google Gemini offers significant advantages over OpenAI:

✅ **Generous Free Tier** - No billing setup required initially  
✅ **Multiple Free Models** - Gemini 2.0 Flash, 2.5 Flash, 1.5 Flash all FREE  
✅ **No Credit Card Required** - Start using immediately  
✅ **Higher Rate Limits** - More requests per minute than OpenAI free tier  
✅ **Advanced Multimodal** - Better at understanding context and medical scenarios  
✅ **Latest Technology** - Gemini 2.0 Flash is Google's newest and most capable model  

## 📋 Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ installed
2. **Google Account**: You'll need a Google account to access Gemini API
3. **Internet Connection**: Required for API calls

## 🔑 Step 1: Get Your Free Google Gemini API Key

### Option A: Quick Setup (Recommended)
1. **Visit Google AI Studio**: Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. **Sign in** with your Google account
3. **Create API Key**: Click "Create API key" button
4. **Copy the Key**: Your key will start with "AIza..." - copy it immediately
5. **Save Securely**: Store it safely (you won't be able to see it again)

### Option B: Through Google Cloud Console
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the "Generative Language API"
4. Create credentials (API Key)
5. Copy your API key

## 🚀 Step 2: Install Dependencies

```bash
# Install the Google Generative AI library
pip install google-generativeai==0.8.3

# Or install all requirements
pip install -r requirements.txt
```

## ⚙️ Step 3: Configure the Application

### Method A: Environment Variable (Recommended)
```bash
# Set environment variable (Windows)
set GEMINI_API_KEY=your_api_key_here

# Set environment variable (Linux/Mac)
export GEMINI_API_KEY=your_api_key_here
```

### Method B: Web Interface Setup
1. **Start the Application**:
   ```bash
   python app.py
   ```

2. **Navigate to Chat Assessment**: Go to the Professional Cardiovascular Consultation

3. **Click "🤖 AI Assistant"** button in the chat interface

4. **Enter Your API Key**: 
   - Paste your Gemini API key (starts with "AIza...")
   - Click "Set API Key"
   - You'll see "AI Enhancement Activated!" message

## 🧪 Step 4: Test the Integration

### Quick Test
1. **Start a Conversation**: Type "Hello" in the chat
2. **Check for AI Response**: You should see intelligent, contextual responses
3. **Test Medical Queries**: Try "I have chest pain" to see medical assessment

### Verification
- **AI Active Indicator**: The AI button should show "AI Active" and turn green
- **Intelligent Responses**: Responses should be detailed and medically relevant
- **Context Awareness**: The AI should remember previous parts of the conversation

## 📊 Understanding Gemini Models

### Available Models (All FREE!)

| Model | Best For | Context Window | Speed |
|-------|----------|----------------|--------|
| **Gemini 2.0 Flash** ⭐ | Balanced performance, medical consultation | 1M tokens | Fast |
| **Gemini 2.5 Flash** | Reasoning, complex medical scenarios | 1M tokens | Fast |
| **Gemini 2.5 Pro** | Most advanced, complex cases | 2M tokens | Moderate |
| **Gemini 1.5 Flash** | Quick responses, simple queries | 1M tokens | Very Fast |

*Current implementation uses **Gemini 2.0 Flash** for optimal performance.*

## 💰 Pricing Information

### Free Tier Benefits
- **No billing setup required**
- **15 requests per minute** per model
- **1,500 requests per day** per model
- **FREE context caching** up to 1M tokens
- **FREE Google Search grounding** (500 requests/day)

### Usage Estimates
- **Typical consultation**: 10-20 requests
- **Daily capacity**: 75+ full consultations
- **Cost**: **$0.00** for normal usage

### If You Exceed Free Limits
- Enable billing for higher rate limits
- Very affordable: ~$0.001-0.01 per consultation
- Much cheaper than OpenAI

## 🛠️ Troubleshooting

### Common Issues

**1. "Invalid API key" Error**
```
Solution: 
- Ensure key starts with "AIza"
- Check for extra spaces or characters
- Regenerate key if needed
```

**2. "API not configured" Error**
```
Solution:
- Enable Generative Language API in Google Cloud Console
- Wait 2-3 minutes for propagation
```

**3. "Rate limit exceeded" Error**
```
Solution:
- Wait 1 minute and try again
- Enable billing for higher limits (optional)
- Use during off-peak hours
```

**4. Import Error**
```bash
# Install missing dependencies
pip install google-generativeai python-dotenv
```

### Debug Mode
Enable debug logging by adding to your environment:
```bash
export GEMINI_DEBUG=true
```

## 🔒 Security Best Practices

### API Key Security
1. **Never commit API keys** to version control
2. **Use environment variables** for production
3. **Rotate keys regularly** (every 90 days)
4. **Restrict API key access** in Google Cloud Console

### Application Security
- Keys are stored only in session/environment
- No permanent storage of API keys
- Secure HTTPS communication with Google APIs

## 🩺 Medical Features

### Enhanced Capabilities
- **Professional Consultation Flow**: Structured medical interview
- **Symptom Analysis**: Intelligent pattern recognition
- **Severity Assessment**: Mild, moderate, severe classifications
- **Emergency Detection**: Immediate alerts for serious symptoms
- **Context Awareness**: Remembers conversation history
- **Medical Language**: Professional terminology with patient-friendly explanations

### Safety Features
- **Emergency Detection**: Bypasses AI for immediate emergency responses
- **Disclaimer Integration**: Appropriate medical disclaimers
- **Professional Guidelines**: Follows medical consultation best practices

## 📈 Performance Optimization

### Response Speed
- **Gemini 2.0 Flash**: ~1-2 seconds response time
- **Context Caching**: Faster subsequent responses
- **Efficient Prompting**: Optimized for medical consultations

### Quality Enhancement
- **Temperature**: 0.7 (balanced creativity/accuracy)
- **Top-p**: 0.8 (focused responses)
- **Max tokens**: 500 (concise but complete)

## 🆕 Latest Features

### Gemini 2.0 Flash Capabilities
- **Multimodal Understanding**: Text, images, audio
- **1M Token Context**: Long conversation memory
- **Agent Mode**: Multi-step reasoning
- **Live API**: Real-time interaction
- **Code Execution**: Dynamic computation

## 🔄 Migration from OpenAI

If migrating from OpenAI:
1. **Replace API key**: Use Gemini key instead of OpenAI
2. **Update endpoints**: All handled automatically
3. **Improved responses**: Generally better medical understanding
4. **Cost savings**: Free tier much more generous

## 📞 Support and Resources

### Documentation
- **Official Docs**: [https://ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
- **API Reference**: [https://ai.google.dev/api](https://ai.google.dev/api)
- **Community**: [Google AI Forum](https://discuss.ai.google.dev/)

### Getting Help
1. **Check console logs** for error messages
2. **Verify API key** is correctly configured
3. **Test with simple queries** first
4. **Check Google Cloud Console** for API status

## ✅ Quick Setup Checklist

- [ ] Get Google Gemini API key from AI Studio
- [ ] Install `google-generativeai` package
- [ ] Set API key via web interface or environment variable
- [ ] Test with "Hello" message
- [ ] Verify AI responses are working
- [ ] Test medical query like "chest pain"
- [ ] Confirm emergency detection works

## 🎉 Congratulations!

**Your Google Gemini-powered cardiovascular consultation system is ready!** 

You now have access to:
- 🤖 **Advanced AI medical consultation**
- 🆓 **Generous free tier**
- ⚡ **Fast, intelligent responses**
- 🛡️ **Enterprise-grade safety**
- 🌟 **Latest AI technology**

**Start helping patients with intelligent, AI-enhanced cardiovascular assessments!** 🩺✨

---

*Last updated: 2025-01-16*  
*Gemini API Version: 0.8.3*  
*Model: Gemini 2.0 Flash* 