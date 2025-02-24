import requests
from typing_extensions import Self
from typing import TypedDict
from promptflow.tracing import trace


class ModelEndpoints:
    def __init__(self: Self, env: dict, model_type: str) -> str:
        self.env = env
        self.model_type = model_type

    class Response(TypedDict):
        query: str
        response: str

    @trace
    def __call__(self: Self, query: str) -> Response:
        if self.model_type == "gpt-4o":
            output = self.call_gpt4o_endpoint(query)
        elif self.model_type == "gpt-4o-mini":
            output = self.call_gpt4omini_endpoint(query)
        elif self.model_type == "mistral7b":
            output = self.call_mistral_endpoint(query)
        elif self.model_type == "tiny_llama":
            output = self.call_tiny_llama_endpoint(query)
        elif self.model_type == "Phi-4":
            output = self.call_phi4(query)
        elif self.model_type == "gpt2":
            output = self.call_gpt2_endpoint(query)
        else:
            output = self.call_default_endpoint(query)

        return output

    def query(self: Self, endpoint: str, headers: str, payload: str) -> str:
        response = requests.post(url=endpoint, headers=headers, json=payload)
        return response.json()

    def call_gpt4o_endpoint(self: Self, query: str) -> Response:
        endpoint = self.env["gpt-4o"]["endpoint"]
        key = self.env["gpt-4o"]["key"]

        headers = {"Content-Type": "application/json", "api-key": key}

        payload = {"messages": [{"role": "user", "content": query}], "max_tokens": 500}

        output = self.query(endpoint=endpoint, headers=headers, payload=payload)
        response = output["choices"][0]["message"]["content"]
        return {"query": query, "response": response}

    def call_gpt4omini_endpoint(self: Self, query: str) -> Response:
        endpoint = self.env["gpt-4o-mini"]["endpoint"]
        print("Endpoint: ", endpoint)
        key = self.env["gpt-4o-mini"]["key"]
        print("Key: ", key)

        headers = {"Content-Type": "application/json", "api-key": key}

        payload = {"messages": [{"role": "user", "content": query}], "max_tokens": 500}
        print("Payload: ", payload)

        output = self.query(endpoint=endpoint, headers=headers, payload=payload)
        print("Output: ", output)
        response = output["choices"][0]["message"]["content"]
        return {"query": query, "response": response}

    def call_tiny_llama_endpoint(self: Self, query: str) -> Response:
        endpoint = self.env["tiny_llama"]["endpoint"]
        key = self.env["tiny_llama"]["key"]

        headers = {"Content-Type": "application/json", "Authorization": ("Bearer " + key)}

        payload = {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 500,
            "stream": False,
        }

        output = self.query(endpoint=endpoint, headers=headers, payload=payload)
        response = output["choices"][0]["message"]["content"]
        return {"query": query, "response": response}

    def call_phi4(self: Self, query: str) -> Response:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential

        endpoint = self.env["phi-4"]["endpoint"]
        key = self.env["phi-4"]["key"]

        client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

        output = client.complete(
        messages=[
            # SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=query)
        ],
        model = "Phi-4",
        max_tokens=1000
        )

        response = output["choices"][0]["message"]["content"]
        return {"query": query, "response": response}

    def call_gpt2_endpoint(self: Self, query: str) -> Response:
        endpoint = self.env["gpt2"]["endpoint"]
        key = self.env["gpt2"]["key"]

        headers = {"Content-Type": "application/json", "Authorization": ("Bearer " + key)}

        payload = {
            "inputs": query,
        }

        output = self.query(endpoint=endpoint, headers=headers, payload=payload)
        response = output[0]["generated_text"]
        return {"query": query, "response": response}

    def call_mistral_endpoint(self: Self, query: str) -> Response:
        endpoint = self.env["mistral7b"]["endpoint"]
        key = self.env["mistral7b"]["key"]

        headers = {"Content-Type": "application/json", "Authorization": ("Bearer " + key)}

        payload = {"messages": [{"content": query, "role": "user"}], "max_tokens": 50}

        output = self.query(endpoint=endpoint, headers=headers, payload=payload)
        response = output["choices"][0]["message"]["content"]
        return {"query": query, "response": response}

    def call_default_endpoint(self: Self, query: str) -> Response:
        return {"query": query, "response": "Paris"}
