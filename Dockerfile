FROM python:3.11-slim

WORKDIR /app

# Instala as dependências essenciais do sistema e os pacotes Python em uma única camada segura
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas numpy pyyaml scikit-learn mlflow scikit-skops && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copia o restante do código do projeto
COPY . .

# Comando padrão para rodar o treino do MLP
CMD ["python", "-m", "src.train", "--model", "mlp"]