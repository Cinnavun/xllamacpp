import os
import sys
import json
import time
import base64
import pytest
import requests
import threading
from typing import Dict, Any

import xllamacpp as xlc


class TestServerHTTP:
    """Test suite for xllamacpp HTTP server endpoints"""

    @pytest.fixture(scope="class")
    def server_url(self):
        """Start HTTP server using xllamacpp.Server and return base URL"""
        # Configure server parameters
        params = xlc.CommonParams()
        params.model.path = os.path.join(
            os.path.dirname(__file__), "../models/Llama-3.2-1B-Instruct-Q8_0.gguf"
        )
        params.n_parallel = 1
        params.n_ctx = 256
        params.cpuparams.n_threads = 2
        params.cpuparams_batch.n_threads = 2
        params.endpoint_metrics = True
        params.sleep_idle_seconds = 1  # Set sleep time to 1 second for testing

        # Create server instance - this automatically starts the HTTP server
        server = xlc.Server(params)

        # Wait for server to be ready - default port is likely 8080
        base_url = server.listening_address
        max_wait = 5  # seconds
        wait_interval = 0.5

        for _ in range(int(max_wait / wait_interval)):
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code == 200:
                    yield base_url
                    break
            except requests.exceptions.RequestException:
                time.sleep(wait_interval)
        else:
            pytest.fail("Server failed to start within timeout period")

        # Server will be automatically cleaned up when the object goes out of scope

    def test_health_endpoints(self, server_url):
        """Test health check endpoints"""
        # Test /health
        response = requests.get(f"{server_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

        # Test /v1/health
        response = requests.get(f"{server_url}/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_models_endpoints(self, server_url):
        """Test model listing endpoints"""
        # Test /models
        response = requests.get(f"{server_url}/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"

        # Test /v1/models
        response = requests.get(f"{server_url}/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_props_endpoints(self, server_url):
        """Test server properties endpoints"""
        # Test GET /props
        response = requests.get(f"{server_url}/props")
        assert response.status_code == 200
        data = response.json()
        assert "build_info" in data

    def test_metrics_endpoint(self, server_url):
        """Test metrics endpoint"""
        response = requests.get(f"{server_url}/metrics")
        assert response.status_code == 200
        # Metrics should be in Prometheus format
        assert "llamacpp:" in response.text

    def test_completion_endpoints(self, server_url):
        """Test text completion endpoints"""
        completion_data = {
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.1,
        }

        # Test /v1/completions (OpenAI compatible)
        response = requests.post(f"{server_url}/v1/completions", json=completion_data)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_chat_completion_endpoints(self, server_url):
        """Test chat completion endpoints"""
        chat_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "max_tokens": 10,
            "temperature": 0.1,
        }

        # Test /chat/completions
        response = requests.post(f"{server_url}/chat/completions", json=chat_data)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

        # Test /v1/chat/completions (OpenAI compatible)
        response = requests.post(f"{server_url}/v1/chat/completions", json=chat_data)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_tokenize_endpoints(self, server_url):
        """Test tokenization endpoints"""
        tokenize_data = {"content": "Hello world, how are you?"}

        # Test /tokenize
        response = requests.post(f"{server_url}/tokenize", json=tokenize_data)
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data

        # Test /detokenize
        detokenize_data = {"tokens": [1, 2, 3, 4, 5]}
        response = requests.post(f"{server_url}/detokenize", json=detokenize_data)
        assert response.status_code == 200
        data = response.json()
        assert "content" in data

    def test_apply_template_endpoint(self, server_url):
        """Test template application endpoint"""
        template_data = {
            "messages": [
                {"role": "system", "content": "You are a test."},
                {"role": "user", "content": "Hi there"},
            ]
        }

        response = requests.post(f"{server_url}/apply-template", json=template_data)
        assert response.status_code == 200
        body = response.json()
        assert "prompt" in body
        assert "You are a test." in body["prompt"]

    def test_slots_endpoints(self, server_url):
        """Test slots management endpoints"""
        # Test GET /slots
        response = requests.get(f"{server_url}/slots")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_streaming_completions(self, server_url):
        """Test streaming completion endpoints"""
        completion_data = {
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "stream": True,
        }

        response = requests.post(
            f"{server_url}/completions", json=completion_data, stream=True
        )
        assert response.status_code == 200

        # Read streaming response
        lines = response.iter_lines()
        first_line = next(lines, None)
        assert first_line is not None
        assert first_line.startswith(b"data: ")

    def test_streaming_chat_completions(self, server_url):
        """Test streaming chat completion endpoints"""
        chat_data = {
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "max_tokens": 10,
            "stream": True,
        }

        response = requests.post(
            f"{server_url}/chat/completions", json=chat_data, stream=True
        )
        assert response.status_code == 200

        # Read streaming response
        lines = response.iter_lines()
        first_line = next(lines, None)
        assert first_line is not None
        assert first_line.startswith(b"data: ")

    def test_error_handling(self, server_url):
        """Test error handling for invalid requests"""
        # Test invalid JSON
        response = requests.post(f"{server_url}/completions", data="invalid json")
        assert response.status_code == 500

        # Test missing required fields
        response = requests.post(f"{server_url}/completions", json={})
        assert response.status_code in [400, 422]

        # Test invalid endpoint
        response = requests.get(f"{server_url}/invalid_endpoint")
        assert response.status_code == 404

    def test_concurrent_requests(self, server_url):
        """Test handling of concurrent requests"""
        import concurrent.futures

        def make_request():
            response = requests.get(f"{server_url}/health")
            return response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]

        # All requests should succeed
        assert all(status == 200 for status in results)

    def test_server_sleep(self, server_url):
        """Test server sleep functionality with sleep_idle_seconds parameter.

        This test verifies that:
        1. Server goes to sleep after idle time
        2. Health and props endpoints remain responsive during sleep
        3. Server reports is_sleeping=True when asleep
        4. Generation request wakes up the server
        5. Server reports is_sleeping=False when awake

        Based on: thirdparty/llama.cpp/tools/server/tests/unit/test_sleep.py
        """
        # Wait a bit so that server can go to sleep
        time.sleep(2)

        # Make sure these endpoints are still responsive after sleep
        response = requests.get(f"{server_url}/health")
        assert response.status_code == 200

        response = requests.get(f"{server_url}/props")
        assert response.status_code == 200
        assert response.json()["is_sleeping"] == True

        # Make a generation request to wake up the server
        response = requests.post(
            f"{server_url}/completion",
            json={
                "n_predict": 1,
                "prompt": "Hello",
            },
        )
        assert response.status_code == 200

        # It should no longer be sleeping
        response = requests.get(f"{server_url}/props")
        assert response.status_code == 200
        assert response.json()["is_sleeping"] == False

    def test_server_sleep_python_api_wake(self, server_url):
        """Test that Python API methods can wake up the server from sleep.

        This test verifies that:
        1. Server goes to sleep after idle time
        2. Python API methods (handle_completions) can wake up the server
        3. Server reports is_sleeping=False after Python API call
        """
        # Wait a bit so that server can go to sleep
        time.sleep(2)

        # Verify server is sleeping
        response = requests.get(f"{server_url}/props")
        assert response.status_code == 200
        assert response.json()["is_sleeping"] == True

        # Now use the same server instance (via HTTP) to wake it up
        # Make a completion request via HTTP to wake up the server
        response = requests.post(
            f"{server_url}/completion",
            json={
                "n_predict": 1,
                "prompt": "Hello",
            },
        )
        assert response.status_code == 200

        # Verify server is now awake
        response = requests.get(f"{server_url}/props")
        assert response.status_code == 200
        assert response.json()["is_sleeping"] == False

        # Wait for server to go to sleep again
        time.sleep(2)

        # Verify server is sleeping again
        response = requests.get(f"{server_url}/props")
        assert response.status_code == 200
        assert response.json()["is_sleeping"] == True

        # Now test that we can also wake it up via chat completion
        response = requests.post(
            f"{server_url}/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
            },
        )
        assert response.status_code == 200

        # Verify server is awake again
        response = requests.get(f"{server_url}/props")
        assert response.status_code == 200
        assert response.json()["is_sleeping"] == False


# Test with embedding model
class TestServerHTTPEmbedding:
    """Test suite for embedding-specific HTTP endpoints"""

    @pytest.fixture(scope="class")
    def embedding_server_url(self):
        """Start HTTP server using xllamacpp.Server with embedding model"""
        # Configure server parameters for embedding model
        params = xlc.CommonParams()
        params.model.path = os.path.join(
            os.path.dirname(__file__), "../models/Qwen3-Embedding-0.6B-Q8_0.gguf"
        )
        params.embedding = True
        params.n_predict = -1
        params.n_ctx = 512
        params.n_batch = 128
        params.n_ubatch = 128
        params.cpuparams.n_threads = 2
        params.cpuparams_batch.n_threads = 2
        params.pooling_type = xlc.llama_pooling_type.LLAMA_POOLING_TYPE_LAST

        # Create server instance - this automatically starts the HTTP server
        server = xlc.Server(params)

        # Wait for server to be ready - use different port to avoid conflicts
        base_url = server.listening_address
        max_wait = 5
        wait_interval = 0.5

        for _ in range(int(max_wait / wait_interval)):
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code == 200:
                    yield base_url
                    break
            except requests.exceptions.RequestException:
                time.sleep(wait_interval)
        else:
            pytest.fail("Embedding server failed to start within timeout period")

        # Server will be automatically cleaned up when the object goes out of scope

    def test_embedding_model_specific(self, embedding_server_url):
        """Test embedding-specific functionality"""
        embedding_data = {
            "input": [
                "I believe the meaning of life is",
                "This is a test",
                "This is another test",
            ]
        }

        response = requests.post(
            f"{embedding_server_url}/v1/embeddings", json=embedding_data
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3

        # Check embedding dimensions (should be consistent)
        first_embedding = data["data"][0]["embedding"]
        assert len(first_embedding) > 0

        for item in data["data"]:
            assert len(item["embedding"]) == len(first_embedding)


# Test with rerank model
class TestServerHTTPRerank:
    """Test suite for rerank-specific HTTP endpoints"""

    @pytest.fixture(scope="class")
    def rerank_server_url(self):
        """Start HTTP server using xllamacpp.Server with rerank model"""
        # Configure server parameters for rerank model
        params = xlc.CommonParams()
        params.model.path = os.path.join(
            os.path.dirname(__file__), "../models/bge-reranker-v2-m3-Q2_K.gguf"
        )
        params.embedding = True
        params.n_predict = -1
        params.n_ctx = 512
        params.n_batch = 128
        params.n_ubatch = 128
        params.cpuparams.n_threads = 2
        params.cpuparams_batch.n_threads = 2
        params.pooling_type = xlc.llama_pooling_type.LLAMA_POOLING_TYPE_RANK

        # Create server instance - this automatically starts the HTTP server
        server = xlc.Server(params)

        # Wait for server to be ready - use different port to avoid conflicts
        base_url = server.listening_address
        max_wait = 5
        wait_interval = 0.5

        for _ in range(int(max_wait / wait_interval)):
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code == 200:
                    yield base_url
                    break
            except requests.exceptions.RequestException:
                time.sleep(wait_interval)
        else:
            pytest.fail("Rerank server failed to start within timeout period")

        # Server will be automatically cleaned up when the object goes out of scope

    def test_rerank_model_specific(self, rerank_server_url):
        """Test rerank-specific functionality"""
        TEST_DOCUMENTS = [
            "A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
            "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
            "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
            "Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine.",
        ]

        response = requests.post(
            f"{rerank_server_url}/rerank",
            json={
                "query": "Machine learning is",
                "documents": TEST_DOCUMENTS,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 4

        most_relevant = body["results"][0]
        least_relevant = body["results"][0]
        for doc in body["results"]:
            if doc["relevance_score"] > most_relevant["relevance_score"]:
                most_relevant = doc
            if doc["relevance_score"] < least_relevant["relevance_score"]:
                least_relevant = doc

        assert most_relevant["relevance_score"] > least_relevant["relevance_score"]
        assert most_relevant["index"] == 2
        assert least_relevant["index"] == 3
