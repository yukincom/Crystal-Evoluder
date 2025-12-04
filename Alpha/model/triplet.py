# トリプレット用

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import openai

class TripletExtractor:
    def __init__(self, mode: str, model_name: str, api_key: str = None):
        self.mode = mode

        if mode == "local":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif mode == "api":
            openai.api_key = api_key
            self.model_name = model_name
        else:
            raise ValueError("mode must be 'local' or 'api'")

    def extract(self, text):
        if self.mode == "local":
            inputs = self.tokenizer(text, return_tensors="pt")
            out = self.model.generate(**inputs, max_new_tokens=256)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)

        if self.mode == "api":
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": f"Extract triples from: {text}"}],
                timeout=90
            )
            return response["choices"][0]["message"]["content"]
