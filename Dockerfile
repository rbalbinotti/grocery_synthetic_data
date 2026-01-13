# Use Python 3.11 como especificado no pyproject.toml
FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de configuração primeiro (para cache de camadas)
COPY ./pyproject.toml ./README.md ./

# Instalar PDM (gerenciador de pacotes Python)
RUN pip install pdm

# Configurar PDM para instalar dependências no sistema
RUN pdm config python.use_venv false

# Instalar dependências do projeto
RUN pdm install --check --prod

# Copiar o restante do código da aplicação
COPY . .

# Criar um usuário não-root para segurança
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Comando padrão (pode ser sobrescrito)
CMD ["python", "--version"]