#!/bin/bash
# Crystal Cluster UI起動スクリプト

echo "💎 Crystal Cluster UI 起動中..."

# 現在のディレクトリを確認
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 仮想環境の確認（オプション）
if [ -d "venv" ]; then
    echo "仮想環境を有効化..."
    source venv/bin/activate
fi

# 依存関係チェック
echo "依存関係を確認中..."
python3 -c "import streamlit" 2>/dev/null || {
    echo "❌ Streamlitがインストールされていません"
    echo "インストールしますか？ (y/n)"
    read -r answer
    if [ "$answer" = "y" ]; then
        pip install streamlit
    else
        exit 1
    fi
}

# Neo4jの確認
echo "Neo4jの接続を確認中..."
nc -z localhost 7687 2>/dev/null || {
    echo "⚠️  Neo4jが起動していません"
    echo "Neo4jを起動してください："
    echo "  docker start neo4j"
    echo ""
    echo "続行しますか？ (y/n)"
    read -r answer
    if [ "$answer" != "y" ]; then
        exit 1
    fi
}

# Ollamaの確認（オプショナル）
echo "Ollamaの接続を確認中（オプション）..."
curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && {
    echo "✅ Ollama接続OK"
} || {
    echo "ℹ️  Ollama未起動（APIモード利用時は不要）"
}

echo ""
echo "========================================="
echo "  💎 Crystal Cluster UI"
echo "========================================="
echo ""
echo "ブラウザが自動で開きます..."
echo "開かない場合は以下にアクセス："
echo "  http://localhost:8501"
echo ""
echo "終了: Ctrl+C"
echo ""

# Streamlit起動
cd frontend
streamlit run app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false