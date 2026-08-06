import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
#from pprint import pprint

# ==========================
# Configurações
# ==========================
MODEL_NAME = "meta/llama-3.1-70b-instruct"
TEMPERATURE = 0.2
MAX_TOKENS = 500

HISTORY_FILE = "chat_history.json"

Message = Dict[str, str]

SYSTEM_PROMPT = (
    "Você é um assistente útil e didático. "
    "Responda em português de forma clara, objetiva e com exemplos quando fizer sentido. "
    "Se a pergunta for ambígua, faça 1 ou 2 perguntas de esclarecimento."
)

# ==========================
# Carrega API Key
# ==========================

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

# ==========================
# Cliente NVIDIA
# ==========================

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

# ==========================
# Agentes
# ==========================

def nova_convercacao() -> List[Message]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

def chatbot_agente(tipo_agente: str) -> str:
    agentes = {
        "atendimento": (
            "Você é um atendente virtual educado, objetivo e profissional."
        ),
        "vendas": (
            "Você é um consultor de vendas. Ajude o cliente a escolher o melhor produto."
        ),
        "suporte": (
            "Você é um técnico de suporte. Resolva problemas técnicos passo a passo."
        )
    }

    return agentes.get(tipo_agente, SYSTEM_PROMPT)

def save_history(messages: List[Message], path: str = HISTORY_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def load_history(path: str = HISTORY_FILE) -> List[Message]:
    if not os.path.exists(path):
        return nova_convercacao()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return nova_convercacao()

        return data

    except Exception:
        return nova_convercacao()

# ==========================
# Chamada da API
# ==========================

def ask_nvidia(messages: List[Message]) -> str:

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return completion.choices[0].message.content

# ==========================
# Programa Principal
# ==========================

messages = load_history()

print("=" * 60)
print("Chatbot NVIDIA")
print(f"Modelo: {MODEL_NAME}")
print("Comandos:")
print("/chatbot")
print("/sair")
print("=" * 60)

while True:

    user_text = input("\nVocê: ").strip()

    if not user_text:
        continue

    comando = user_text.lower()

    if comando == "/sair":
        print("Até logo!")
        break

    if comando == "/chatbot":

        print("\nAgentes disponíveis:")
        print("1 - Atendimento")
        print("2 - Vendas")
        print("3 - Suporte")

        opcao = input("Escolha: ")

        mapa = {
            "1": "atendimento",
            "2": "vendas",
            "3": "suporte"
        }

        tipo = mapa.get(opcao)
        prompt = chatbot_agente(tipo)

        messages = [
            {
                "role": "system",
                "content": prompt
            }
        ]

        print(f"Agente '{tipo}' ativado.")
        continue

    # De volta ao chat normal
    messages.append(
        {
            "role": "user",
            "content": user_text
        }
    )

    try:
        resposta = ask_nvidia(messages)

        messages.append(
            {
                "role": "assistant",
                "content": resposta
            }
        )

        save_history(messages)

        print(f"\nAssistente: {resposta}")

    except Exception as e:
        print(f"Ocorreu um erro: {e}")

    try:

        resposta = ask_nvidia(messages)

        messages.append(
            {
                "role": "assistant",
                "content": resposta,
            }
        )

        save_history(messages)

        print(f"\nAssistente: {resposta}")
    except Exception as e:
        print(f"Ocorreu um erro ao chamar a API: {e}")