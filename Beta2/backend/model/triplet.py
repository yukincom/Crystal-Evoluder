# トリプレット用
import re
from typing import List, Tuple
import requests
import openai

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class TripletExtractor:
    """
    トリプレット抽出用AI
    
    Modes:
        - "api": OpenAI API（gpt-4o-mini, claude-3.5-sonnet等）
        - "ollama": Ollama（llama3.1:14B等）
        - "local": Hugging Face Transformers（Rebel, Qwen2.5等）
    """
    
    def __init__(
        self,
        mode: str,
        model_name: str,
        api_key: str = None,
        device: str = "mps",
        ollama_url: str = "http://localhost:11434"
    ):
        """
        Args:
            mode: "api" または "local"
            model_name: 
                - API: "gpt-4o-mini", "claude-3.5-sonnet"
                - Ollama: "llama3.1:14B", "qwen2.5:14B"
                - Local: "Babelscape/rebel-large", "Qwen/Qwen2.5-14B"
            api_key: OpenAI API key（API mode時）
            device: "mps", "cuda", "cpu"
            ollama_url: Ollama URL
        """
        self.mode = mode
        self.model_name = model_name
        self.ollama_url = ollama_url 
        
        if mode == "ollama":  
            print(f"🏠  Ollama mode: {model_name} at {ollama_url}")
            self._check_ollama()

        if mode == "local":
            print(f"📦 Loading local model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # GPU/MPS対応
            if device == "mps" and hasattr(self.model, "to"):
                self.model.to("mps")
            elif device == "cuda":
                self.model.to("cuda")
            
            print(f"✅ Local model loaded on {device}")
        
        elif mode == "api":
            if not api_key:
                raise ValueError("API key is required for API mode")
            
            openai.api_key = api_key
            print(f"✅ API mode: {model_name}")
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'api' or 'local'")
    
    def _check_ollama(self):  # 👈 追加
        """Ollamaが起動しているか確認"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✅ Ollama is running at {self.ollama_url}")
            else:
                print(f"⚠️  Ollama responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"❌ Cannot connect to Ollama at {self.ollama_url}\n"
                "Please start Ollama:\n"
                "  ollama serve\n"
                f"  ollama pull {self.model_name}"
            )
            
    def extract(self, text: str, max_triplets: int = 15) -> List[Tuple[str, str, str]]:
        """
        テキストからトリプレットを抽出
        
        Args:
            text: 入力テキスト
            max_triplets: 最大トリプレット数
        
        Returns:
            [(subject, relation, object), ...] のリスト
        """
        if self.mode == "ollama": 
            return self._extract_ollama(text, max_triplets)
        elif self.mode == "local":
            return self._extract_local(text, max_triplets)
        elif self.mode == "api":
            return self._extract_api(text, max_triplets)
    
    def _extract_ollama(self, text: str, max_triplets: int) -> List[Tuple]:  # 👈 追加
        """Ollama経由で抽出"""
        prompt = f"""Extract knowledge graph triples from the following text.
Return up to {max_triplets} triples in JSON array format: [["subject", "relation", "object"], ...]

Text: {text}

Return ONLY the JSON array, no explanation:"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": 0.3,
                    "stream": False
                },
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            content = result.get('response', '')
            
            # JSONパース
            triplets = self._parse_json_output(content)
            return triplets[:max_triplets]
        
        except Exception as e:
            print(f"⚠️  Ollama extraction failed: {e}")
            return []
    def _extract_local(self, text: str, max_triplets: int) -> List[Tuple]:
        """ローカルモデルで抽出"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3
        )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # パース（モデル依存）
        triplets = self._parse_rebel_output(decoded)
        return triplets[:max_triplets]
    
    def _extract_api(self, text: str, max_triplets: int) -> List[Tuple]:
        """API経由で抽出"""
        prompt = f"""Extract knowledge graph triples from the following text.
Return up to {max_triplets} triples in the format: (subject, relation, object)

Text: {text}

Triples:"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                timeout=90,
                temperature=0.1
            )
            
            content = response["choices"][0]["message"]["content"]
            triplets = self._parse_llm_output(content)
            return triplets[:max_triplets]
        
        except Exception as e:
            print(f"⚠️  API extraction failed: {e}")
            return []
    
    def _parse_json_output(self, text: str) -> List[Tuple]:  # 👈 追加
        """JSON形式の出力をパース"""
        import json
        
        # JSONを抽出（前後の説明文を除去）
        # ```json ... ``` 形式に対応
        text = text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        try:
            triplets_list = json.loads(text)
            
            # リストの各要素をタプルに変換
            result = []
            for item in triplets_list:
                if isinstance(item, list) and len(item) == 3:
                    result.append((str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()))
            
            return result
        
        except json.JSONDecodeError:
            print(f"⚠️  Failed to parse JSON: {text[:100]}...")
            return []
    
    def _parse_rebel_output(self, text: str) -> List[Tuple]:
        """Rebel モデル出力のパース"""
        # 実装例（簡易版）
        triplets = []
        # ... パースロジック ...
        return triplets
    
    def _parse_llm_output(self, text: str) -> List[Tuple]:
        """LLM出力のパース"""
        
        triplets = []
        
        # (subject, relation, object) 形式を探す
        pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        matches = re.findall(pattern, text)
        
        for s, r, o in matches:
            triplets.append((s.strip(), r.strip(), o.strip()))
        
        return triplets