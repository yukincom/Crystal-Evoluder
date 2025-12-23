# Crystal Cluster

Knowledge Graph RAG System with React Frontend and FastAPI Backend.

## Project Structure

```
crystal_cluster/
├── backend/                    # FastAPI backend
│   ├── main.py                # FastAPI app with CORS
│   ├── core/
│   ├── processors/
│   ├── builders/
│   └── ...
├── frontend/                   # React frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── App.tsx            # Main component
│   │   ├── components/
│   │   │   ├── AIModeSwitcher.tsx
│   │   │   ├── FileUpload.tsx
│   │   ├── api/
│   │   │   └── client.ts      # API client
│   │   └── types/
│   │       └── index.ts       # Type definitions
│   └── public/
├── docker-compose.yml          # Docker compose for both services
└── README.md
```

## Development Setup

### Backend (FastAPI)
```bash
cd backend
uvicorn main:app --reload
```

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

### Using Docker Compose
```bash
docker-compose up --build
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /ai_status` - Get AI mode and status
- `POST /switch_mode` - Switch AI mode (api/ollama)
- `POST /upload` - Upload files for processing

## Ports

- Backend: http://localhost:8000
- Frontend: http://localhost:3000