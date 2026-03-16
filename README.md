# EVA - Personal AI Assistant Platform

EVA is a personal AI assistant platform designed for web, desktop, and mobile. It provides a conversational AI companion with emotional awareness, memory capabilities, and interactive modes.

## Architecture Overview

```
eva-backend/
├── main.py                  # FastAPI application entry point
├── config.py                # Settings management (env vars)
├── modes.py                 # Assistant mode system (Chat/Game)
├── database.py              # Firestore database manager
├── auth.py                  # JWT authentication logic
├── auth_router.py           # Auth API endpoints (login, register, Firebase)
├── firebase_auth.py         # Firebase Auth for Google login
├── conversation_handler.py  # Conversation processing orchestration
├── context_window.py        # LLM context window management
├── memory_manager.py        # Memory system (core, event, conversational)
├── memory_extractor.py      # Memory detection from conversations
├── llm_service.py           # Google Gemini LLM integration
├── api.py                   # Conversation REST endpoints + modes
├── api_memory.py            # Memory management endpoints
├── api_sync.py              # Offline sync endpoints
├── api_tools.py             # Tool/function call definitions
├── websocket_manager.py     # WebSocket chat handler
├── schemas.py               # Pydantic request/response schemas
├── models.py                # Core data models
├── exceptions.py            # Custom exception classes
├── security.py              # Security headers middleware
├── rate_limiter.py          # API rate limiting
├── cache_manager.py         # In-memory TTL caching
├── error_middleware.py       # Error handling middleware
├── logging_config.py        # Structured logging
├── utils.py                 # Utility functions
├── secrets_router.py        # Secrets management endpoints
├── settings.py              # Extended settings
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
├── frontend/                # Web frontend
│   ├── index.html           # Landing page + chat interface
│   ├── css/style.css        # Styles
│   └── js/
│       ├── config.js        # Firebase + API configuration
│       ├── auth.js          # Google sign-in handling
│       └── chat.js          # Chat interface logic
└── tests/                   # Test suite
    ├── conftest.py          # Test fixtures
    └── test_*.py            # Test modules
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python / FastAPI |
| Database | Google Cloud Firestore |
| Authentication | Firebase Auth (Google login) |
| AI/LLM | Google Gemini API |
| Hosting (backend) | Google Cloud Run |
| Hosting (frontend) | Firebase Hosting |
| Frontend | Vanilla HTML/CSS/JS |

## Backend Modules

### Authentication (`auth.py`, `auth_router.py`, `firebase_auth.py`)
- JWT token-based auth for API requests
- Firebase Auth integration for Google sign-in
- User registration and password management
- Multi-method authentication (JWT, Firebase, Cloud Run headers)

### Conversation System (`conversation_handler.py`, `api.py`)
- Processes user messages through the LLM
- Manages conversation context and history
- Handles function/tool calling
- Supports mode switching (Chat, Game)

### Memory System (`memory_manager.py`, `memory_extractor.py`, `api_memory.py`)
- **Core memories**: Persistent facts about the user (name, preferences, etc.)
- **Event memories**: Time-based events with expiration
- **Conversational memories**: Context from past conversations
- Automatic memory detection from natural language
- Memory management endpoints (CRUD)

### Mode System (`modes.py`)
- **Chat Mode**: Standard AI conversation with memory and tools
- **Game Mode**: Interactive story/game experience (stub for future)
- Each mode has its own system prompt and features
- Mode switching via API

### LLM Integration (`llm_service.py`)
- Google Gemini API integration
- Streaming response support
- Function/tool calling
- Mock mode for development without API key

### Tool System (`api_tools.py`)
- Extensible tool framework
- Built-in tools: Time, Weather, Memory
- Tools can be called by the LLM during conversation

### Database (`database.py`)
- Firestore operations for all collections
- In-memory fallback for development
- Conversation storage with message subcollections

## Firestore Collections

| Collection | Description |
|------------|-------------|
| `users` | User accounts (id, email, username, role, preferences) |
| `memories` | User memories (core facts, events, conversation memories) |
| `conversations` | Conversation sessions with messages subcollection |
| `api_keys` | API key records |
| `categories` | Secret categories |
| `secrets` | Encrypted user secrets |

## API Routes

### Authentication (`/api/v1/auth`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/register` | Register a new user |
| POST | `/login` | Login with username/password |
| POST | `/firebase` | Login with Firebase/Google token |
| GET | `/me` | Get current user details |
| POST | `/refresh-token` | Refresh access token |
| POST | `/logout` | Log out |
| POST | `/change-password` | Change password |
| GET | `/verify-token` | Verify token validity |

### Conversation (`/api/v1/conversation`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/` | Send message (supports mode switching) |
| GET | `/modes` | Get available assistant modes |

### Memory (`/api/v1/memory`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/core` | Create core memory |
| POST | `/event` | Create event memory |
| GET | `/{memory_id}` | Get specific memory |
| GET | `/query/` | Search memories |
| PUT | `/{memory_id}` | Update memory |
| DELETE | `/{memory_id}` | Delete memory |
| POST | `/text/extract` | Extract memory from text |

### WebSocket (`/ws`)
| Path | Description |
|------|-------------|
| `/chat` | Real-time streaming chat |

### Other
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/docs` | Swagger UI |
| GET | `/api/redoc` | ReDoc documentation |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | Yes | - | Secret key for JWT signing |
| `GEMINI_API_KEY` | Yes* | - | Google Gemini API key (*mock mode if absent) |
| `APP_ENV` | No | `development` | Environment (development/staging/production) |
| `PORT` | No | `8080` | Server port |
| `FIREBASE_PROJECT_ID` | No | - | Firebase/GCP project ID |
| `FIREBASE_CREDENTIALS_PATH` | No | `/app/secrets/firebase-credentials.json` | Path to Firebase credentials |
| `USE_GCP_DEFAULT_CREDENTIALS` | No | `false` | Use GCP Application Default Credentials |
| `GEMINI_MODEL` | No | `gemini-1.5-flash-latest` | Gemini model to use |
| `LLM_TEMPERATURE` | No | `0.7` | LLM response temperature |
| `LLM_MAX_TOKENS` | No | `2048` | Max tokens for LLM responses |
| `CONTEXT_MAX_TOKENS` | No | `2048` | Max tokens in context window |
| `CORS_ORIGINS` | No | `*` | Allowed CORS origins (comma-separated) |
| `BACKEND_URL` | No | Auto-detect | Backend URL for token audience |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `DEBUG` | No | `false` | Enable debug mode |

## Running Locally

### Prerequisites
- Python 3.11+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/IAmLep/eva-backend.git
   cd eva-backend
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   # Create a .env.development file
   echo 'SECRET_KEY=your-dev-secret-key-change-in-production' > .env.development
   echo 'GEMINI_API_KEY=your-gemini-api-key' >> .env.development
   echo 'APP_ENV=development' >> .env.development
   ```

5. **Configure the frontend** (optional - for Google login)
   - Create a Firebase project at https://console.firebase.google.com
   - Enable Google sign-in in Firebase Auth
   - Update `frontend/js/config.js` with your Firebase project settings
   - Download the Firebase Admin SDK credentials JSON
   - Set `FIREBASE_CREDENTIALS_PATH` to point to the credentials file

6. **Run the server**
   ```bash
   python main.py
   ```
   The server starts at `http://localhost:8080`

7. **Access the application**
   - Frontend: http://localhost:8080
   - API docs: http://localhost:8080/api/docs
   - Health check: http://localhost:8080/health

### Running Tests
```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

## Deployment

### Google Cloud Run

1. **Build the Docker image**
   ```bash
   docker build -t eva-backend .
   ```

2. **Push to Google Container Registry**
   ```bash
   docker tag eva-backend gcr.io/YOUR_PROJECT_ID/eva-backend
   docker push gcr.io/YOUR_PROJECT_ID/eva-backend
   ```

3. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy eva-backend \
     --image gcr.io/YOUR_PROJECT_ID/eva-backend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars "SECRET_KEY=your-secret,GEMINI_API_KEY=your-key"
   ```

### Firebase Hosting (Frontend)

For separate frontend hosting:

1. Install Firebase CLI: `npm install -g firebase-tools`
2. Initialize: `firebase init hosting` (set `frontend/` as public directory)
3. Deploy: `firebase deploy --only hosting`

## Development Phases

### Phase 1 (Current) ✅
- [x] Backend API with FastAPI
- [x] Firebase Auth (Google login)
- [x] Chat interface (web)
- [x] Conversation processing with Gemini
- [x] Firestore conversation storage
- [x] Memory system (detection + storage)
- [x] Tool system (time, weather, memory)
- [x] Mode switching architecture
- [x] EVA personality/system prompt

### Phase 2 (Planned)
- [ ] Game Mode implementation
- [ ] Voice interaction (push-to-talk)
- [ ] Memory management UI
- [ ] Conversation history in sidebar
- [ ] Enhanced emotional awareness

### Phase 3 (Future)
- [ ] Live voice conversation
- [ ] Mobile application
- [ ] Desktop application
- [ ] Advanced game scenarios
- [ ] Vector embeddings for memory search

## License

Private project. All rights reserved.
