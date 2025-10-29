# Smart Investment Planner

An intelligent investment recommendation system that analyzes user financial profiles and risk appetite to provide personalized portfolio allocations and AI-driven financial advice using local LLM (Ollama) and predictive models.

## 🎯 Key Features

- **ML-Powered Allocation**: Uses Random Forest to predict optimal investment allocation (SIP, FD, Stocks)
- **Local AI Integration**: Generates personalized advice using Ollama (llama3.1:8b) running locally
- **RAG System**: Retrieval-Augmented Generation for financial knowledge-based responses
- **Stock Price Prediction**: ML-based stock price forecasting with technical analysis
- **Interactive Analytics**: Comprehensive dashboard with charts and portfolio visualization
- **Data Privacy**: All computations and AI interactions processed locally
- **Real-time Predictions**: Instant investment recommendations based on user profile

## Project Structure

```
rag-investor/
├── backend/                 # Flask API server
│   ├── app.py              # Main Flask application
│   ├── ml_train.py         # ML model training
│   ├── utils/              # Utility modules
│   │   ├── ml_utils.py     # ML prediction functions
│   │   ├── rag_utils.py    # RAG system
│   │   └── report.py       # PDF report generation
│   ├── data/               # Data storage
│   │   ├── kb/             # Knowledge base files
│   │   └── uploads/        # Generated reports
│   └── requirements.txt    # Python dependencies
├── frontend/               # React application
│   ├── src/
│   │   ├── components/     # React components
│   │   └── api.js          # API client
│   └── package.json        # Node.js dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Option 1: One-Click System Start (Windows)

**Start Complete System**:
```bash
start_system.bat
```
This will start both backend and frontend servers automatically.

### Option 2: Individual Components

1. **Start Backend**:
   ```bash
   start_backend.bat
   ```

2. **Start Frontend** (in a new terminal):
   ```bash
   start_frontend.bat
   ```

### Option 3: Manual Setup

#### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd rag-investor/backend
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup script**:
   ```bash
   python setup.py
   ```

4. **Start the Flask server**:
   ```bash
   python app.py
   ```

The backend will be available at `http://127.0.0.1:5500`

#### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd rag-investor/frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Start the React development server**:
   ```bash
   npm start
   ```

The frontend will be available at `http://localhost:3000`

## Usage

1. **Open the application** in your browser at `http://localhost:3000`
2. **Fill in your financial details**:
   - Monthly Income
   - Monthly Expenses
   - Age
   - Risk Tolerance (1-5 scale)
3. **Click "Get Recommendation"** to receive:
   - ML-predicted investment allocation
   - Personalized financial advice
4. **View your history** and download reports

## API Endpoints

- `POST /predict` - Get investment recommendation
- `GET /history` - Get user's prediction history
- `GET /report/<id>` - Download PDF report

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///app.db
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b
API_PORT=5500
```

### Ollama Setup (Required for AI Advice)

**Ollama is essential for AI-powered investment advice generation:**

1. **Install Ollama**: Download from https://ollama.ai/
2. **Pull the required model**:
   ```bash
   ollama pull llama3.1:8b
   ```
3. **Start Ollama service**:
   ```bash
   ollama serve
   ```
4. **Verify installation**:
   ```bash
   ollama list
   ```

**Important**: The system requires Ollama to be running for AI advice generation. Without it, only basic fallback advice will be provided.

## Troubleshooting

### Common Issues

1. **"No advice available"**:
   - Check if Ollama is running
   - The system will use fallback advice if Ollama is unavailable

2. **ML model errors**:
   - Run `python setup.py` to retrain the model
   - Check if `ml_model.pkl` exists in the backend directory

3. **Import errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

4. **Frontend not connecting to backend**:
   - Ensure backend is running on port 5500
   - Check CORS settings in `app.py`

### Dependencies

**Backend**:
- Flask 2.3.3
- scikit-learn 1.3.0
- sentence-transformers 2.2.2
- pandas 2.0.3
- numpy 1.24.3

**Frontend**:
- React 19.1.1
- Axios 1.12.2
- Chart.js 4.5.0
- Tailwind CSS 4.1.13

## Development

### Adding New Knowledge Base Content

1. Add `.txt` files to `backend/data/kb/`
2. The RAG system will automatically index new content
3. Restart the backend to rebuild the index

### Modifying ML Model

1. Edit `ml_train.py` to change model parameters
2. Run `python ml_train.py` to retrain
3. The new model will be used automatically

### Customizing Advice

1. Modify `get_fallback_advice()` in `app.py`
2. Update the prompt template for Ollama
3. Add new knowledge base files for better context

## License

This project is for educational and personal use only.
