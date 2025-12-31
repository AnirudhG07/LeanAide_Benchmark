import concurrent.futures
import json
from pathlib import Path

import requests


class BenchmarkRunner:
    """
    A class to automate the process of benchmarking the LeanAide formalization pipeline.
    """
    def __init__(self, website_url, provider, model, workers=4):
        """
        Initializes the BenchmarkRunner.

        Args:
            website_url (str): The base URL of the LeanAide web application.
            provider (str): The LLM provider (e.g., 'openai').
            model (str): The LLM model name (e.g., 'gpt-4').
            workers (int): The number of parallel workers to use for processing problems.
        """
        self.website_url = website_url.rstrip('/')
        self.llm_provider = provider
        self.llm_model = model
        self.max_workers = workers

        # API endpoints
        self.generate_proof_url = f"{self.website_url}/api/llm/generate_proof"
        self.generate_json_url = f"{self.website_url}/api/llm/generate_json"
        self.leanaide_api_url = f"{self.website_url}/api/leanaide"

    def _get_ai_proof(self, problem):
        """Step 1: Generate AI Proof for a given problem."""
        print(f"  [Problem {problem['id']}] Step 1: Generating AI proof...")
        payload = {
            "theorem": problem["formal_statement"],
            "existingProof": "sorry",
            "provider": self.llm_provider,
            "model": self.llm_model,
        }
        response = requests.post(self.generate_proof_url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["proof"]

    def _get_structured_json(self, problem, ai_proof):
        """Step 2: Generate Structured JSON from the theorem and AI proof."""
        print(f"  [Problem {problem['id']}] Step 2: Generating structured JSON...")
        payload = {
            "mode": "theorem-proof",
            "theorem": problem["formal_statement"],
            "proof": ai_proof,
            "provider": self.llm_provider,
            "model": self.llm_model,
        }
        response = requests.post(self.generate_json_url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()

    def _start_lean_code_generation(self, document_json):
        """Step 3: Start the asynchronous job for Lean code generation."""
        payload = {
            "task": "lean_from_json_structured",
            "document_json": document_json,
            "mode": "async",
        }
        response = requests.post(self.leanaide_api_url, json=payload)
        response.raise_for_status()
        return response.json()

    def process_problem(self, problem):
        """
        Runs the initial processing pipeline for a single problem.
        (Steps: AI Proof -> Structured JSON -> Start Async Job)
        """
        try:
            ai_proof = self._get_ai_proof(problem)
            structured_json = self._get_structured_json(problem, ai_proof)
            async_job_data = self._start_lean_code_generation(structured_json)

            return {
                "id": problem["id"],
                "informal_statement": problem["informal_statement"],
                "original_formal_statement": problem["formal_statement"],
                "ai_proof": ai_proof,
                "structured_json": structured_json,
                "job_token": async_job_data.get("token"),
                "job_status": "running",
                "generated_lean_code": None,
            }
        except Exception as e:
            print(f"  [Problem {problem['id']}] FAILED: {e}")
            return {
                "id": problem["id"],
                "informal_statement": problem["informal_statement"],
                "error": str(e),
            }

    def run_benchmark(self, input_path, output_path):
        """
        Reads an input file and runs the benchmark pipeline in parallel.
        Saves progress incrementally to the output file.
        """
        input_p = Path(input_path)
        output_p = Path(output_path)

        print("Starting benchmark...")
        print(f"  Input file: {input_p.resolve()}")
        print(f"  Output file: {output_p.resolve()}")
        print(f"  Parallel workers: {self.max_workers}")

        with open(input_p, 'r') as f:
            problems = json.load(f)

        # Initialize output file as a JSON list
        output_p.write_text("[\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_problem = {executor.submit(self.process_problem, p): p for p in problems}
            
            is_first_result = True
            for future_idx, future in enumerate(concurrent.futures.as_completed(future_to_problem)):
                problem = future_to_problem[future]
                try:
                    result = future.result()
                    print(f"COMPLETED: Problem {problem['id']}")
                    
                    # Append result to the JSON file incrementally
                    with open(output_p, 'a') as f:
                        if not is_first_result:
                            f.write(",\n")
                        json.dump(result, f, indent=2)
                        is_first_result = False

                except Exception as e:
                    print(f"ERROR processing problem {problem['id']}: {e}")

        # Finalize the JSON list
        # The json.dump with indent=2 already adds a trailing newline,
        # so we just need to add the closing bracket on a new line.
        with open(output_p, 'a') as f:
            f.write("]\n")


        print(f"Benchmark run finished. Results saved to {output_p.resolve()}")

    def update_results(self, results_path):
        """
        Reads a results file, polls for the status of 'running' jobs,
        and updates the file with the final generated Lean code.
        """
        results_p = Path(results_path)
        print(f"Updating results file: {results_p.resolve()}")
        
        with open(results_p, 'r') as f:
            results = json.load(f)

        updated_count = 0
        for result in results:
            if result.get("job_status") == "running" and result.get("job_token"):
                token = result["job_token"]
                print(f"  Checking status for token: {token} (Problem {result['id']})")
                
                try:
                    status_payload = {"mode": "lookup", "token": token}
                    response = requests.post(self.leanaide_api_url, json=status_payload)
                    response.raise_for_status()
                    status_data = response.json()

                    if status_data.get("status") == 0: # Completed
                        print(f"    -> COMPLETED: Problem {result['id']}")
                        lean_code = status_data["data"]["completed"]["result"]["document_code"]
                        result["job_status"] = "completed"
                        result["generated_lean_code"] = lean_code
                        updated_count += 1
                    elif status_data.get("status") == 2: # Error
                        print(f"    -> ERROR: Problem {result['id']}")
                        result["job_status"] = "error"
                        result["error"] = status_data.get("data", "Unknown error during Lean code generation.")
                        updated_count += 1
                    else:
                        print("    -> Still running...")

                except Exception as e:
                    print(f"    -> FAILED to update status for token {token}: {e}")

        if updated_count > 0:
            print(f"Updated {updated_count} job(s). Saving changes to {results_p.resolve()}...")
            with open(results_p, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print("No running jobs completed since last check.")

if __name__ == '__main__':
    # --- Configuration ---
    CONFIG = {
        "website_url": "http://10.134.13.103:3000",
        "provider": "openai",
        "model": "gpt-5.1",
        "workers": 4, # Number of parallel threads
    }

    # --- File Paths ---
    INPUT_FILE = "fate_1.json"
    OUTPUT_FILE = "benchmark_results.json"

    # --- Script Usage ---
    runner = BenchmarkRunner(**CONFIG)
    runner.run_benchmark(INPUT_FILE, OUTPUT_FILE)

