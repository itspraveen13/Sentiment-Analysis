# 🎯 Sentimetryx - AI-Powered Sentiment Analysis Platform

**Transform Customer Feedback into Actionable Insights**

Sentimetryx is an enterprise-grade sentiment analysis platform that leverages artificial intelligence, deep learning, and natural language processing to provide comprehensive business intelligence tools. Empower your organization to enhance customer retention strategies and drive data-driven decisions.

---

## ✨ Key Features

### 📊 **1. Sentiment Analysis**

Analyze customer sentiment with three flexible methods:

- **Text Input** - Quick analysis of individual reviews or feedback
- **Dataset Upload** - Batch process CSV files with hundreds of reviews
- **Audio Analysis** - Extract sentiment from call recordings and voice feedback

### 🔮 **2. Churn Prediction**

Identify at-risk customers before they leave. Predict customer churn probability and receive early warning signals to improve retention.

### 👂 **3. Social Listening**

Monitor social media conversations, track brand mentions, and analyze sentiment across platforms. Stay informed about what customers are saying about your brand.

### ⚠️ **4. Fake Review Detection**

Automatically identify and filter fraudulent reviews. Ensure your feedback metrics reflect genuine customer experiences.

### 😂 **5. Meme Analysis**

Analyze sentiment in visual content. Understand emotions expressed through memes combining images and text.

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. **Clone and navigate to the project**

   ```bash
   cd Sentiment-Analysis
   ```

2. **Install dependencies**

   ```bash
   npm install --legacy-peer-deps
   ```

3. **Configure environment** (first time only)

   ```bash
   cp .env.example .env
   ```

4. **Start the application**

   ```bash
   npm start
   ```

5. **Access the application**
   - 🎨 **Frontend**: http://localhost:3000/
   - 💻 **Backend API**: http://localhost:5000/
   - ✅ **Health Check**: http://localhost:5000/health

That's it! The app is now running. 🎉

---

## 📖 Usage Guide

### Starting the Application

**Start Everything (Recommended)**

```bash
npm start
```

Runs backend and frontend servers simultaneously.

**Start Individual Components**

```bash
npm run backend      # Backend API only
npm run frontend     # Frontend UI only
```

For more commands, see [NPM_COMMANDS_CHEATSHEET.md](./NPM_COMMANDS_CHEATSHEET.md)

---

## 🛠️ Technology Stack

### Frontend

- **React 18** - Interactive UI components
- **Vite** - Fast development and production builds
- **Material-UI** - Professional UI components
- **Chart.js** - Data visualization
- **Axios** - API communication

### Backend

- **Node.js** - JavaScript runtime
- **Express** - Web server framework
- **Sentiment Analysis** - Keyword-based analysis
- **File Upload** - CSV and audio processing

### Optional: Python Backend

- **Flask** - REST API framework
- **TensorFlow** - Deep learning models
- **BERT** - Advanced NLP models
- **PRAW** - Reddit data extraction
- **scikit-learn** - Machine learning algorithms

---

## 📁 Project Structure

```
Sentiment-Analysis/
├── 📚 Documentation
│   ├── README.md                    # This file
│   ├── START_HERE.md               # Quick start guide
│   ├── SETUP.md                    # Detailed setup
│   ├── CONFIG_QUICK_REFERENCE.md   # Configuration options
│   └── NPM_COMMANDS_CHEATSHEET.md  # Command reference
│
├── 🔧 Configuration
│   ├── .env                        # Environment variables
│   ├── package.json                # Project metadata & scripts
│   ├── vite.config.js              # Vite configuration
│   └── backend-constants.js        # Backend configuration
│
├── 💻 Backend
│   ├── backend.js                  # Node.js API server
│   └── Backend/                    # Optional Python backend
│       ├── app.py                  # Flask API
│       ├── sentiment.py            # Sentiment analysis
│       ├── churn.py                # Churn prediction
│       ├── fake.py                 # Fake review detection
│       └── requirements.txt        # Python dependencies
│
└── 🎨 Frontend
    ├── src/
    │   ├── App.jsx                 # Main application
    │   ├── components/             # React components
    │   ├── pages/                  # Page components
    │   └── assets/                 # Images and styles
    └── public/                     # Static files
```

---

## ⚙️ Configuration

### Environment Variables

The application uses a `.env` file for configuration. Key settings:

```env
# Server Configuration
BACKEND_PORT=5000
NODE_ENV=development

# API Settings
VITE_API_BASE_URL=http://localhost:5000
VITE_API_TIMEOUT=10000

# File Upload
MAX_FILE_SIZE=10485760        # 10MB
MAX_ROWS_PER_ANALYSIS=100     # Max CSV rows

# Sentiment Analysis
CONFIDENCE_THRESHOLD=0.5
MIN_CONFIDENCE_SCORE=0.6
MAX_CONFIDENCE_SCORE=0.99
```

For detailed configuration options, see [CONFIG_QUICK_REFERENCE.md](./CONFIG_QUICK_REFERENCE.md)

---

## 📋 Available Commands

| Command            | Purpose                             |
| ------------------ | ----------------------------------- |
| `npm start`        | Run everything (frontend + backend) |
| `npm run backend`  | Backend API only                    |
| `npm run frontend` | Frontend UI only                    |
| `npm run build`    | Production build                    |
| `npm run preview`  | Preview production build            |

---

## 🎥 Demos

Experience Sentimetryx in action:

- **[3-Minute Demo](https://youtu.be/5omLQWjJSqo)** - Quick overview of key features
- **[10-Minute Demo](https://youtu.be/eGZ38wzJEMA)** - Comprehensive feature walkthrough

---

## 🔌 API Endpoints

### Sentiment Analysis

```
POST /text          # Analyze single text
POST /dataset       # Analyze CSV file
POST /audio         # Analyze audio file
```

### Predictions & Analysis

```
POST /churn         # Predict customer churn
POST /fake          # Detect fake reviews
POST /social        # Analyze social media
POST /meme          # Analyze meme sentiment
```

### System

```
GET /health         # Check API status
```

---

## 📊 Example Usage

### Analyze Text

```bash
curl -X POST http://localhost:5000/text \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```

### Check API Health

```bash
curl http://localhost:5000/health
```

---

## 🔧 Development

### First Time Setup

```bash
npm install --legacy-peer-deps
cp .env.example .env
npm start
```

### Making Changes

1. Edit source files in `src/` (frontend) or `backend.js` (backend)
2. Changes automatically reload during development
3. Check http://localhost:3000 to see updates

### Production Build

```bash
npm run build
npm run preview  # Test production build locally
```

---

## 📚 Documentation

For detailed information, see:

| Document                                                   | Purpose               |
| ---------------------------------------------------------- | --------------------- |
| [START_HERE.md](./START_HERE.md)                           | Quick overview        |
| [SETUP.md](./SETUP.md)                                     | Complete setup guide  |
| [CONFIG_QUICK_REFERENCE.md](./CONFIG_QUICK_REFERENCE.md)   | Configuration options |
| [NPM_COMMANDS_CHEATSHEET.md](./NPM_COMMANDS_CHEATSHEET.md) | All npm commands      |
| [CLEANUP_SUMMARY.md](./CLEANUP_SUMMARY.md)                 | What was improved     |
| [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md)         | Complete docs index   |

---

## 🐛 Troubleshooting

### Problem: Port Already in Use

**Solution:** Change the port in `.env`

```env
BACKEND_PORT=5001
```

### Problem: API Connection Failed

**Solution:** Verify the backend is running and check `.env`

```bash
curl http://localhost:5000/health
```

### Problem: Dependencies Installation Failed

**Solution:** Use legacy peer deps flag

```bash
npm install --legacy-peer-deps
```

### Problem: Need More Help?

See [SETUP.md](./SETUP.md) for comprehensive troubleshooting guide.

---

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is part of Team TechAnz's initiative to democratize AI-powered business intelligence.

---

## 👥 Team

**Developed by:** Team TechAnz  
**Project Name:** Sentimetryx  
**Vision:** Transform customer feedback into actionable business intelligence

---

## 📞 Support

- 📖 Check the [documentation](./DOCUMENTATION_INDEX.md)
- 🐛 Report issues on GitHub
- 💬 Reach out to the team

---

## ✅ Getting Started Checklist

- [ ] Read this README
- [ ] Install dependencies: `npm install --legacy-peer-deps`
- [ ] Configure `.env`: `cp .env.example .env`
- [ ] Start the app: `npm start`
- [ ] Open http://localhost:3000
- [ ] Try analyzing some text!

---

**Ready to get started? Run `npm start` now!** 🚀
