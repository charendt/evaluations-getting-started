from typing import TypedDict, Self
from promptflow.tracing import trace


class ModelEndpoints:
    def __init__(self: Self, model: dict) -> str:
        self.model = model

    class Response(TypedDict):
        query: str
        response: str

    @trace
    def __call__(self: Self, query: str) -> Response:
        output = self.chat_completion(query)
        return output

    def chat_completion(self: Self, query: str) -> Response:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential
        from dotenv import load_dotenv
        import os

        endpoint = os.getenv("AZURE_OPENAI_INFERENCE_ENDPOINT")
        key = os.getenv("AZURE_OPENAI_API_KEY")

        print(f"endpoint: {endpoint}")
        print(f"model: {self.model} \n")

        client = ChatCompletionsClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key),
            model=self.model,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.95,
        )

        output = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=query)
            ]
        )

        response = output["choices"][0]["message"]["content"]
        return {"query": query, "response": response}