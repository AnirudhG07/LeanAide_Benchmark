import concurrent.futures
import json
from pathlib import Path

import requests


class BenchmarkRunner:
    """
    A class to automate the process of benchmarking the LeanAide formalization pipeline.
    It supports resuming from intermediate steps and parallel execution.
    """
    def __init__(self, website_url, provider, model, workers=4):
        self.website_url = website_url.rstrip('/')
        self.llm_provider = provider
        self.llm_model = model
        self.max_workers = workers

        self.generate_proof_url = f"{self.website_url}/api/llm/generate_proof"
        self.generate_json_url = f"{self.website_url}/api/llm/generate_json"
        self.leanaide_api_url = f"{self.website_url}/api/leanaide"

    def _get_ai_proof(self, problem):
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
        data = response.json()
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                print("  [ERROR] Failed to parse JSON string from _get_structured_json")
                raise
        return data

    def _start_lean_code_generation(self, document_json, problem):
        print(f"  [Problem {problem['id']}] Step 3: Starting Lean code generation job...")
        payload = {
            "task": "lean_from_json_structured",
            "document_json": document_json,
            "mode": "async",
            "source": problem.get("source"),
            "problem_id": problem.get("id"),
        }
        response = requests.post(self.leanaide_api_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def _update_problem_status(self, result):
        if result.get("job_status") == "running" and result.get("job_token"):
            token = result["job_token"]
            print(f"  [Problem {result['id']}] Updating status for token: {token}")
            
            try:
                status_payload = {"mode": "lookup", "token": token}
                response = requests.post(self.leanaide_api_url, json=status_payload)
                response.raise_for_status()
                status_data = response.json()

                if status_data.get("status") == 0:
                    print("    -> COMPLETED")
                    lean_code = status_data.get("data", {}).get("result", {}).get("document_code")
                    result["job_status"] = "completed"
                    result["generated_lean_code"] = lean_code
                elif status_data.get("status") == 2:
                    print("    -> ERROR")
                    result["job_status"] = "error"
                    result["error"] = status_data.get("data", "Unknown error during Lean code generation.")
                else:
                    print("    -> Still running...")
            except Exception as e:
                print(f"    -> FAILED to update status for token {token}: {e}")
        return result

    def process_problem(self, problem, existing_result=None):
        """
        Runs the full pipeline for a single problem, resuming from where it left off.
        """
        result = existing_result or { "id": problem["id"], "informal_statement": problem["informal_statement"], "original_formal_statement": problem["formal_statement"] }

        try:
            if "ai_proof" not in result:
                result["ai_proof"] = self._get_ai_proof(problem)

            if "structured_json" not in result:
                result["structured_json"] = self._get_structured_json(problem, result["ai_proof"])
            
            if isinstance(result.get("structured_json"), str):
                 result["structured_json"] = json.loads(result["structured_json"])

            if "job_token" not in result and "structured_json" in result:
                async_data = self._start_lean_code_generation(result["structured_json"], problem)
                result["job_token"] = async_data.get("token")
                result["job_status"] = "running"
            
            return result
        except Exception as e:
            print(f"  [Problem {problem['id']}] FAILED: {e}")
            result["error"] = str(e)
            return result

    def run_benchmark(self, input_path, output_path):
        input_p = Path(input_path)
        output_p = Path(output_path)

        print("Starting benchmark run...")
        print(f"  Input file: {input_p.resolve()}")
        print(f"  Output file: {output_p.resolve()}")

        existing_results = {}
        if output_p.exists():
            try:
                with open(output_p, 'r') as f:
                    results_list = json.load(f)
                    existing_results = {res['id']: res for res in results_list if 'id' in res}
                print(f"  Loaded {len(existing_results)} existing results.")
            except (json.JSONDecodeError, FileNotFoundError):
                print("  Could not read existing results file. Starting fresh.")

        with open(input_p, 'r') as f:
            problems = {p['id']: p for p in json.load(f)}

        problems_to_process = [p for p_id, p in problems.items() if p_id not in existing_results]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_problem = {executor.submit(self.process_problem, p): p for p in problems_to_process}
            
            for future in concurrent.futures.as_completed(future_to_problem):
                problem = future_to_problem[future]
                try:
                    result = future.result()
                    existing_results[result['id']] = result
                    print(f"  Initial processing complete for new problem {result['id']}")
                except Exception as e:
                    print(f"  ERROR processing new problem {problem['id']}: {e}")

        # Save results incrementally after processing new problems
        sorted_results = [existing_results[p_id] for p_id in problems if p_id in existing_results]
        with open(output_p, 'w') as f:
            json.dump(sorted_results, f, indent=2)

        print("Initial processing complete. Now checking status of running jobs...")
        self.update_results(output_path)


    def update_results(self, results_path):
        results_p = Path(results_path)
        if not results_p.exists():
            print(f"Results file not found at {results_p.resolve()}. Please run a benchmark first.")
            return

        print(f"Updating results from: {results_p.resolve()}")
        
        with open(results_p, 'r') as f:
            results = json.load(f)
        
        problems_to_update = [res for res in results if res.get("job_status") == "running"]
        if not problems_to_update:
            print("No running jobs to update.")
            return
            
        print(f"Found {len(problems_to_update)} running jobs to update.")

        updated_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_problem = {executor.submit(self._update_problem_status, p): p for p in problems_to_update}
            
            for future in concurrent.futures.as_completed(future_to_problem):
                updated_result = future.result()
                updated_results.append(updated_result)

        # Merge updated results back into the main list
        final_results_map = {res['id']: res for res in results}
        for res in updated_results:
            final_results_map[res['id']] = res
            
        final_results_list = list(final_results_map.values())
        
        print(f"Saving updated results to {results_p.resolve()}...")
        with open(results_p, 'w') as f:
            json.dump(final_results_list, f, indent=2)
        print("Update complete.")

if __name__ == '__main__':
    CONFIG = {
        "website_url": "http://10.134.13.103:3000",
        "provider": "openai",
        "model": "gpt-5.1", # keep it gpt-5.1
        "workers": 4,
    }
    INPUT_FILE = "fate_1.json"
    OUTPUT_FILE = "benchmark_results.json"

    runner = BenchmarkRunner(**CONFIG)
    
    # --- CHOOSE ACTION ---
    # 1. Run the full pipeline for new problems and then update status of all running jobs.
    runner.run_benchmark(INPUT_FILE, OUTPUT_FILE)
    
    # 2. Only update the status of running jobs in an existing results file.
    # runner.update_results(OUTPUT_FILE)

    print("\nScript finished.")
