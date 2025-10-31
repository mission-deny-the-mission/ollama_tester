#!/usr/bin/env python3
"""
Parallel test suite for Ollama instances and OpenAI-compatible APIs.
Measures time to first token and tokens per second.
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import argparse


@dataclass
class TestMetrics:
    """Stores metrics for a single test run."""
    server_name: str
    endpoint: str
    success: bool
    time_to_first_token: Optional[float] = None  # in seconds
    tokens_per_second: Optional[float] = None
    total_time: Optional[float] = None  # in seconds
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class APITester:
    """Async tester for OpenAI-compatible APIs."""
    
    def __init__(self, timeout: int = 300):
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_ollama(
        self,
        server_name: str,
        base_url: str,
        model: str,
        prompt: str,
        stream: bool = True
    ) -> TestMetrics:
        """Test an Ollama instance."""
        metrics = TestMetrics(
            server_name=server_name,
            endpoint=base_url
        )
        
        url = f"{base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        start_time = time.time()
        first_token_time = None
        token_count = 0
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    metrics.success = False
                    metrics.error_message = f"HTTP {response.status}: {error_text}"
                    return metrics
                
                if stream:
                    async for line in response.content:
                        if not line.strip():
                            continue
                        
                        try:
                            data = json.loads(line)
                            
                            if first_token_time is None:
                                first_token_time = time.time()
                                metrics.time_to_first_token = first_token_time - start_time
                            
                            if "response" in data:
                                token_count += len(data["response"].split())
                            elif "token" in data:
                                token_count += 1
                            
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                    
                    if first_token_time:
                        total_time = time.time() - start_time
                        metrics.total_time = total_time
                        metrics.total_tokens = token_count
                        if total_time > 0:
                            metrics.tokens_per_second = token_count / total_time
                        metrics.success = True
                    else:
                        metrics.success = False
                        metrics.error_message = "No tokens received"
                else:
                    data = await response.json()
                    first_token_time = time.time()
                    metrics.time_to_first_token = first_token_time - start_time
                    
                    response_text = data.get("response", "")
                    token_count = len(response_text.split())
                    total_time = time.time() - start_time
                    
                    metrics.total_time = total_time
                    metrics.total_tokens = token_count
                    metrics.tokens_per_second = token_count / total_time if total_time > 0 else 0
                    metrics.success = True
                    
        except asyncio.TimeoutError:
            metrics.success = False
            metrics.error_message = f"Request timed out after {self.timeout} seconds"
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
        
        return metrics
    
    async def test_openai_compatible(
        self,
        server_name: str,
        base_url: str,
        model: str,
        prompt: str,
        api_key: Optional[str] = None,
        stream: bool = True
    ) -> TestMetrics:
        """Test an OpenAI-compatible API endpoint."""
        metrics = TestMetrics(
            server_name=server_name,
            endpoint=base_url
        )
        
        url = f"{base_url}/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        start_time = time.time()
        first_token_time = None
        completion_tokens = 0
        prompt_tokens = 0
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    metrics.success = False
                    metrics.error_message = f"HTTP {response.status}: {error_text}"
                    return metrics
                
                if stream:
                    async for line in response.content:
                        if not line.strip():
                            continue
                        
                        line_text = line.decode('utf-8')
                        
                        if line_text.startswith("data: "):
                            line_text = line_text[6:]
                        
                        if line_text.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(line_text)
                            
                            choices = data.get("choices", [])
                            if choices and len(choices) > 0:
                                delta = choices[0].get("delta", {})
                                if "content" in delta and first_token_time is None:
                                    first_token_time = time.time()
                                    metrics.time_to_first_token = first_token_time - start_time
                                
                                if "content" in delta:
                                    content = delta["content"]
                                    completion_tokens += len(content.split())
                            
                            if "usage" in data:
                                usage = data["usage"]
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                completion_tokens = usage.get("completion_tokens", completion_tokens)
                        except json.JSONDecodeError:
                            continue
                    
                    if first_token_time:
                        total_time = time.time() - start_time
                        metrics.total_time = total_time
                        metrics.completion_tokens = completion_tokens
                        metrics.prompt_tokens = prompt_tokens
                        if total_time > 0:
                            metrics.tokens_per_second = completion_tokens / total_time
                        metrics.success = True
                    else:
                        metrics.success = False
                        metrics.error_message = "No tokens received"
                else:
                    data = await response.json()
                    first_token_time = time.time()
                    metrics.time_to_first_token = first_token_time - start_time
                    
                    usage = data.get("usage", {})
                    completion_tokens = usage.get("completion_tokens", 0)
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    
                    total_time = time.time() - start_time
                    
                    metrics.total_time = total_time
                    metrics.completion_tokens = completion_tokens
                    metrics.prompt_tokens = prompt_tokens
                    metrics.tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
                    metrics.success = True
                    
        except asyncio.TimeoutError:
            metrics.success = False
            metrics.error_message = f"Request timed out after {self.timeout} seconds"
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
        
        return metrics


async def run_parallel_tests(config: Dict[str, Any]) -> List[TestMetrics]:
    """Run tests in parallel across all configured servers."""
    test_prompt = config.get("test_prompt", "Write a short story about a robot learning to paint.")
    results = []
    
    async with APITester(timeout=config.get("timeout", 300)) as tester:
        tasks = []
        
        # Create tasks for Ollama instances
        for server_config in config.get("ollama_servers", []):
            server_name = server_config["name"]
            base_url = server_config["base_url"]
            model = server_config["model"]
            stream = server_config.get("stream", True)
            
            task = tester.test_ollama(
                server_name=server_name,
                base_url=base_url,
                model=model,
                prompt=test_prompt,
                stream=stream
            )
            tasks.append(task)
        
        # Create tasks for OpenAI-compatible servers
        for server_config in config.get("openai_servers", []):
            server_name = server_config["name"]
            base_url = server_config["base_url"]
            model = server_config["model"]
            api_key = server_config.get("api_key")
            stream = server_config.get("stream", True)
            
            task = tester.test_openai_compatible(
                server_name=server_name,
                base_url=base_url,
                model=model,
                prompt=test_prompt,
                api_key=api_key,
                stream=stream
            )
            tasks.append(task)
        
        # Run all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error metrics for exceptions
                error_metrics = TestMetrics(
                    server_name=f"Server_{i}",
                    endpoint="unknown",
                    success=False,
                    error_message=str(result)
                )
                processed_results.append(error_metrics)
            else:
                processed_results.append(result)
        
        return processed_results


def print_results(results: List[TestMetrics]):
    """Print test results in a formatted table."""
    print("\n" + "="*120)
    print("TEST RESULTS")
    print("="*120)
    print(f"{'Server':<30} {'Status':<10} {'TTFT (s)':<12} {'Tokens/s':<12} {'Total Tokens':<15} {'Total Time (s)':<15}")
    print("-"*120)
    
    for metrics in results:
        status = "✓ PASS" if metrics.success else "✗ FAIL"
        ttft = f"{metrics.time_to_first_token:.3f}" if metrics.time_to_first_token else "N/A"
        tps = f"{metrics.tokens_per_second:.2f}" if metrics.tokens_per_second else "N/A"
        tokens = str(metrics.total_tokens or metrics.completion_tokens or "N/A")
        total_time = f"{metrics.total_time:.3f}" if metrics.total_time else "N/A"
        
        print(f"{metrics.server_name:<30} {status:<10} {ttft:<12} {tps:<12} {tokens:<15} {total_time:<15}")
        
        if not metrics.success and metrics.error_message:
            print(f"  Error: {metrics.error_message}")
    
    print("="*120)
    
    # Print summary statistics
    successful = [r for r in results if r.success]
    if successful:
        print("\nSUMMARY (Successful tests only):")
        print(f"  Total servers tested: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(results) - len(successful)}")
        
        if any(r.time_to_first_token for r in successful):
            avg_ttft = sum(r.time_to_first_token for r in successful if r.time_to_first_token) / len([r for r in successful if r.time_to_first_token])
            print(f"  Average TTFT: {avg_ttft:.3f}s")
        
        if any(r.tokens_per_second for r in successful):
            avg_tps = sum(r.tokens_per_second for r in successful if r.tokens_per_second) / len([r for r in successful if r.tokens_per_second])
            print(f"  Average Tokens/s: {avg_tps:.2f}")
            print(f"  Best Tokens/s: {max(r.tokens_per_second for r in successful if r.tokens_per_second):.2f}")


def save_results_json(results: List[TestMetrics], output_file: str):
    """Save results to a JSON file."""
    results_dict = []
    for metrics in results:
        results_dict.append({
            "server_name": metrics.server_name,
            "endpoint": metrics.endpoint,
            "success": metrics.success,
            "time_to_first_token": metrics.time_to_first_token,
            "tokens_per_second": metrics.tokens_per_second,
            "total_time": metrics.total_time,
            "total_tokens": metrics.total_tokens,
            "completion_tokens": metrics.completion_tokens,
            "prompt_tokens": metrics.prompt_tokens,
            "error_message": metrics.error_message,
            "timestamp": metrics.timestamp
        })
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test Ollama instances and OpenAI-compatible APIs in parallel")
    parser.add_argument("--config", "-c", default="config.json", help="Path to configuration JSON file")
    parser.add_argument("--output", "-o", help="Path to save JSON results (optional)")
    parser.add_argument("--prompt", "-p", help="Override test prompt")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        print("Please create a config.json file. See README.md for examples.")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return 1
    
    # Override prompt if provided
    if args.prompt:
        config["test_prompt"] = args.prompt
    
    print(f"Running parallel tests against {len(config.get('ollama_servers', [])) + len(config.get('openai_servers', []))} servers...")
    print(f"Test prompt: {config.get('test_prompt', 'N/A')}")
    
    # Run tests
    results = asyncio.run(run_parallel_tests(config))
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.output:
        save_results_json(results, args.output)
    
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    exit(main())

