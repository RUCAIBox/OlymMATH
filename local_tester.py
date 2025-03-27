import json
import os
import time
import datetime
import re
import argparse
from collections import defaultdict
from math_verify import parse, verify
from vllm import LLM, SamplingParams
import numpy as np
import traceback
from transformers import AutoTokenizer

# top_p and min_p are set to 0.95 and 0.

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="OlymMATH Evaluator")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--min", type=int, required=True, help="Start index for problems (inclusive)")
    parser.add_argument("--max", type=int, required=True, help="End index for problems (exclusive)")
    parser.add_argument("--sample", type=int, default=10, help="Number of samples per question")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum tokens for generation")
    parser.add_argument(
        "--dataset", type=str, default="EN-EASY",
        choices=["EN-EASY", "EN-HARD", "ZH-EASY", "ZH-HARD"],
        help="Dataset to use (EN-EASY, EN-HARD, ZH-EASY, ZH-HARD)"
    )
    return parser.parse_args()

# Prompts for Chinese and English
PROMPT_CN = "请逐步推理，并在 \\boxed{} 内给出您的最终答案。\n\n"
PROMPT_EN = "Please reason step by step, and put your final answer within \\boxed{}.\n\n"


def extract_boxed(text):
    """
    Extract content from \boxed{} with support for nested braces
    """
    stack = []
    boxed_contents = []
    i = 0
    start_idx = -1

    while i < len(text):
        if text[i : i + 7] == "\\boxed{" and (i == 0 or text[i - 1] != "\\"):
            if not stack:
                start_idx = i + 7
            stack.append("{")
            i += 7
        elif text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            stack.append("{")
            i += 1
        elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    boxed_contents.append(text[start_idx:i])
                    start_idx = -1
            i += 1
        else:
            i += 1

    # Fallback to regex if the first method fails
    if not boxed_contents:
        pattern = r"\\boxed{((?:[^{}]|{(?:[^{}]|{[^{}]*})*})*?)}"
        matches = list(re.finditer(pattern, text))
        if matches:
            return [matches[-1].group(1)]  # Return the last match

    return boxed_contents


def format_for_math_verify(answer):
    """
    Format the answer for math verification, ensuring it has $ symbols
    """
    if not answer:
        return "$0$"  # Return a default value to avoid empty strings

    # Remove any existing dollar signs and surrounding whitespace
    answer = answer.strip()

    # Clear dollar signs at beginning and end
    if answer.startswith("$"):
        answer = answer[1:]
    if answer.endswith("$"):
        answer = answer[:-1]

    # Remove remaining whitespace
    answer = answer.strip()

    # Ensure content is not empty
    if not answer:
        return "$0$"

    # Add dollar signs
    return f"${answer}$"


def string_compare_answers(extracted, gold):
    """
    Compare answers using string normalization as a fallback when math_verify fails
    """

    # Clean and normalize strings
    def normalize(text):
        if not text:
            return ""
        # Remove all whitespace
        text = re.sub(r"\s+", "", text)
        # Replace common equivalent representations
        text = text.replace("\\frac", "")
        text = text.replace("\\cdot", "*")
        text = text.replace("\\times", "*")
        # Remove all LaTeX commands
        text = re.sub(r"\\[a-zA-Z]+", "", text)
        return text

    normalized_extracted = normalize(extracted)
    normalized_gold = normalize(gold)

    # Direct comparison or check for inclusion
    return (
        normalized_extracted == normalized_gold
        or normalized_gold in normalized_extracted
        or normalized_extracted in normalized_gold
    )


def run_evaluation(args):
    """
    Main evaluation function that handles batch inference and evaluation
    """
    # Get parameters from command line arguments
    MODEL = args.model
    GPUS = args.gpus
    MIN = args.min
    MAX = args.max
    USE_CHAT_TEMPLATE = True

    # Get the new parameters from command line arguments
    SAMPLE = args.sample
    TEMPERATURE = args.temperature
    MAX_TOKENS = args.max_tokens
    DATASET_TYPE = args.dataset

    # Map dataset type to file path
    DATASET = f"data/OlymMATH-{DATASET_TYPE}.jsonl"

    # Create log directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs_{timestamp}_{MODEL.replace('/', '_')}"
    os.makedirs(log_dir, exist_ok=True)

    # Load problems from JSONL file
    print(f"Loading problems from {DATASET}")
    all_problems = []
    with open(DATASET, "r", encoding="utf-8") as f:
        for line in f:
            problem_data = json.loads(line)
            all_problems.append(problem_data)
    print(f"Loaded {len(all_problems)} problems")

    # Take problems from MIN to MAX
    selected_problems = all_problems[MIN:MAX]
    print(
        f"Selected {len(selected_problems)} problems for evaluation (index {MIN} to {MAX-1})"
    )

    # Initialize VLLM and tokenizer
    print(f"Initializing VLLM with model {MODEL} on {GPUS} GPUs")
    llm = LLM(model=MODEL, tensor_parallel_size=GPUS)

    # Initialize tokenizer for chat template
    tokenizer = None
    try:
        print(f"Loading tokenizer for chat template from {MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        if hasattr(tokenizer, "apply_chat_template"):
            print("Chat template is available and will be used")
            USE_CHAT_TEMPLATE = True
        else:
            print("Warning: Tokenizer does not have apply_chat_template method, falling back to direct prompts")
            USE_CHAT_TEMPLATE = False
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to direct prompts")
        USE_CHAT_TEMPLATE = False

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=0.95,
        min_p=0.0,
    )

    # Prepare prompts for Chinese and English
    cn_prompts = []
    en_prompts = []
    problems_data = []

    # Determine if we're using Chinese or English dataset
    is_chinese = DATASET_TYPE.startswith("ZH")

    for problem in selected_problems:
        # Get problem text and answer
        problem_text = problem.get("problem", "")

        # Store problem data for later use
        problems_data.append(
            {
                "unique_id": problem.get("unique_id"),
                "cn_problem": problem_text if is_chinese else "",
                "en_problem": problem_text if not is_chinese else "",
                "answer": problem.get("answer", ""),
                "subject": problem.get("subject", ""),
            }
        )

        # Create prompts for each sample
        for _ in range(SAMPLE):
            if USE_CHAT_TEMPLATE:
                # Apply chat template
                if is_chinese:
                    messages = [{"role": "user", "content": PROMPT_CN + problem_text}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    cn_prompts.append(prompt)
                else:
                    messages = [{"role": "user", "content": PROMPT_EN + problem_text}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    en_prompts.append(prompt)
            else:
                # Direct prompt without chat template
                if is_chinese:
                    cn_prompts.append(PROMPT_CN + problem_text)
                else:
                    en_prompts.append(PROMPT_EN + problem_text)

    # Run batch inference based on dataset language
    if is_chinese:
        print("Running batch inference for Chinese prompts...")
        outputs = llm.generate(cn_prompts, sampling_params)
        results = []

        # Process Chinese results
        for i, output in enumerate(outputs):
            problem_idx = i // SAMPLE
            sample_idx = i % SAMPLE
            problem_data = problems_data[problem_idx]

            response_text = output.outputs[0].text
            output_tokens = len(output.outputs[0].token_ids)

            # Extract boxed answer
            boxed_answers = extract_boxed(response_text)
            extracted_answer = boxed_answers[0] if boxed_answers else ""

            # Save result
            if len(results) <= problem_idx:
                results.append([])

            results[problem_idx].append(
                {
                    "response": response_text,
                    "extracted_answer": extracted_answer,
                    "output_tokens": output_tokens,
                    "sample_idx": sample_idx,
                }
            )
    else:
        print("Running batch inference for English prompts...")
        outputs = llm.generate(en_prompts, sampling_params)
        results = []

        # Process English results
        for i, output in enumerate(outputs):
            problem_idx = i // SAMPLE
            sample_idx = i % SAMPLE
            problem_data = problems_data[problem_idx]

            response_text = output.outputs[0].text
            output_tokens = len(output.outputs[0].token_ids)

            # Extract boxed answer
            boxed_answers = extract_boxed(response_text)
            extracted_answer = boxed_answers[0] if boxed_answers else ""

            # Save result
            if len(results) <= problem_idx:
                results.append([])

            results[problem_idx].append(
                {
                    "response": response_text,
                    "extracted_answer": extracted_answer,
                    "output_tokens": output_tokens,
                    "sample_idx": sample_idx,
                }
            )

    # Save all samples for each problem to log files
    for problem_idx, samples in enumerate(results):
        problem_data = problems_data[problem_idx]
        problem_log = {
            "problem": problem_data["cn_problem"] if is_chinese else problem_data["en_problem"],
            "unique_id": problem_data["unique_id"],
            "answer": problem_data["answer"],
            "subject": problem_data["subject"],
            "samples": [],
        }

        for sample in samples:
            problem_log["samples"].append(
                {
                    "response": sample["response"],
                    "extracted_answer": sample["extracted_answer"],
                    "output_tokens": sample["output_tokens"],
                    "sample_idx": sample["sample_idx"],
                }
            )

        # Save to log file
        with open(
            f"{log_dir}/problem_{problem_data['unique_id']}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(problem_log, f, ensure_ascii=False, indent=2)

    # Evaluate results
    evaluation_results = evaluate_results(
        problems_data, results, log_dir, MODEL, len(selected_problems), is_chinese
    )

    # Update log files with evaluation results
    update_logs_with_evaluation(log_dir, evaluation_results, is_chinese)


def update_logs_with_evaluation(log_dir, evaluation_results, is_chinese):
    """
    Updates log files with evaluation results
    """
    print("Updating log files with evaluation results...")

    # Get the sample size from the first problem's evaluation results
    sample_size = 0
    if evaluation_results and evaluation_results[0] and "sample_results" in evaluation_results[0]:
        sample_size = len(evaluation_results[0]["sample_results"])

    for problem_eval in evaluation_results:
        # Read the existing log file
        log_file = f"{log_dir}/problem_{problem_eval['unique_id']}.json"

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)

            # Add evaluation results
            log_data["evaluation"] = {
                "pass@1": problem_eval["pass@1"],
                f"cons@{sample_size}": problem_eval["cons@SAMPLE"],
                "largest_cluster_size": problem_eval["largest_cluster_size"],
                "consistency_correct": problem_eval["consistency_correct"],
            }

            # Add evaluation results for each sample
            for i, sample_result in enumerate(problem_eval["sample_results"]):
                if i < len(log_data["samples"]):
                    log_data["samples"][i]["is_correct"] = sample_result[
                        "is_correct"
                    ]

            # Write back the updated log
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error updating log file {log_file}: {e}")


def evaluate_results(problems_data, results, log_dir, model, problem_count, is_chinese):
    """
    Evaluate model responses against ground truth answers
    """
    print("Evaluating results...")

    # Get the sample size from the first problem's results
    sample_size = len(results[0]) if results and results[0] else 0

    # Store detailed evaluation results for each problem
    evaluation_results = []

    # Prepare data structures for metrics
    metrics = {
        "total": {
            "correct": 0,
            "total": 0,
            "output_tokens": [],
            "response_lengths": [],
            "correct_output_tokens": [],
            "correct_response_lengths": [],
            "incorrect_output_tokens": [],
            "incorrect_response_lengths": [],
        },
        "by_subject": defaultdict(
            lambda: {
                "correct": 0,
                "total": 0,
                "output_tokens": [],
                "response_lengths": [],
                "correct_output_tokens": [],
                "correct_response_lengths": [],
                "incorrect_output_tokens": [],
                "incorrect_response_lengths": [],
                "consistency_correct": 0,
                "consistency_total": 0,
            }
        ),
        "consistency_correct": 0,
        "consistency_total": 0,
    }

    # Function to process a single problem's samples
    def process_problem_samples(problem_idx, problem_data, samples):
        gold_answer = problem_data["answer"]
        subject = problem_data["subject"]

        # Initialize problem evaluation data
        problem_evaluation = {
            "unique_id": problem_data["unique_id"],
            "subject": subject,
            "sample_results": [],
            "pass@1": 0,
            "cons@SAMPLE": 0,
            "largest_cluster_size": 0,
            "consistency_correct": False,
        }

        # Track correctness for each sample and process pass@1
        correct_count = 0
        for sample in samples:
            extracted_answer = sample["extracted_answer"]
            response_text = sample["response"]
            output_tokens = sample["output_tokens"]
            response_length = len(response_text)

            # Try to verify with math_verify
            is_correct = False
            try:
                formatted_gold = format_for_math_verify(gold_answer)
                formatted_extracted = format_for_math_verify(extracted_answer)

                gold_parsed = parse(formatted_gold)
                extracted_parsed = parse(formatted_extracted)

                is_correct = verify(gold_parsed, extracted_parsed)
            except Exception as e:
                # Fallback to string comparison
                try:
                    is_correct = string_compare_answers(extracted_answer, gold_answer)
                except Exception as inner_e:
                    print(f"Error in string comparison: {inner_e}")
                    is_correct = False

            # Save sample evaluation result
            problem_evaluation["sample_results"].append(
                {
                    "is_correct": is_correct,
                    "output_tokens": output_tokens,
                    "response_length": response_length,
                }
            )

            # Record metrics for pass@1
            metrics["total"]["output_tokens"].append(output_tokens)
            metrics["total"]["response_lengths"].append(response_length)
            metrics["by_subject"][subject]["output_tokens"].append(output_tokens)
            metrics["by_subject"][subject]["response_lengths"].append(
                response_length
            )

            if is_correct:
                correct_count += 1
                metrics["total"]["correct"] += 1
                metrics["by_subject"][subject]["correct"] += 1
                metrics["total"]["correct_output_tokens"].append(output_tokens)
                metrics["total"]["correct_response_lengths"].append(
                    response_length
                )
                metrics["by_subject"][subject]["correct_output_tokens"].append(
                    output_tokens
                )
                metrics["by_subject"][subject]["correct_response_lengths"].append(
                    response_length
                )
            else:
                metrics["total"]["incorrect_output_tokens"].append(output_tokens)
                metrics["total"]["incorrect_response_lengths"].append(
                    response_length
                )
                metrics["by_subject"][subject]["incorrect_output_tokens"].append(
                    output_tokens
                )
                metrics["by_subject"][subject][
                    "incorrect_response_lengths"
                ].append(response_length)

            metrics["total"]["total"] += 1
            metrics["by_subject"][subject]["total"] += 1

        # Calculate problem pass@1
        problem_evaluation["pass@1"] = correct_count / len(samples) if samples else 0

        # Process consistency (cons@SAMPLE)
        # Group answers that are equivalent to each other
        answer_clusters = []
        formatted_answers = []

        # Format all extracted answers for comparison
        for sample in samples:
            extracted = sample["extracted_answer"]
            try:
                formatted = format_for_math_verify(extracted)
                parsed = parse(formatted)
                formatted_answers.append((extracted, parsed))
            except Exception as e:
                # If parsing fails, use the raw answer
                formatted_answers.append((extracted, None))

        # Form clusters of equivalent answers
        for i, (raw_answer_i, parsed_i) in enumerate(formatted_answers):
            # Skip if this answer is already in a cluster
            if any(i in cluster for cluster in answer_clusters):
                continue

            # Start a new cluster
            new_cluster = [i]

            # Find all equivalent answers
            for j, (raw_answer_j, parsed_j) in enumerate(formatted_answers):
                if i != j and not any(j in cluster for cluster in answer_clusters):
                    try:
                        # Check if answers are equivalent
                        if parsed_i is not None and parsed_j is not None:
                            if verify(parsed_i, parsed_j):
                                new_cluster.append(j)
                        elif raw_answer_i == raw_answer_j:
                            # Fallback to string comparison
                            new_cluster.append(j)
                    except Exception:
                        # If verification fails, only add if the raw answers are identical
                        if raw_answer_i == raw_answer_j:
                            new_cluster.append(j)

            answer_clusters.append(new_cluster)

        # Find the largest cluster
        largest_cluster = max(answer_clusters, key=len) if answer_clusters else []
        problem_evaluation["largest_cluster_size"] = (
            len(largest_cluster) if largest_cluster else 0
        )

        # Use the first answer from the largest cluster as the consistent answer
        if largest_cluster:
            consistent_answer_idx = largest_cluster[0]
            consistent_answer = samples[consistent_answer_idx]["extracted_answer"]

            # Check if the consistent answer is correct
            try:
                formatted_gold = format_for_math_verify(gold_answer)
                formatted_consistent = format_for_math_verify(consistent_answer)

                gold_parsed = parse(formatted_gold)
                consistent_parsed = parse(formatted_consistent)

                is_consistent_correct = verify(gold_parsed, consistent_parsed)
            except Exception:
                # Fallback to string comparison
                try:
                    is_consistent_correct = string_compare_answers(
                        consistent_answer, gold_answer
                    )
                except Exception:
                    is_consistent_correct = False

            # Save consistency evaluation
            problem_evaluation["cons@SAMPLE"] = 1 if is_consistent_correct else 0
            problem_evaluation["consistency_correct"] = is_consistent_correct

            # Update consistency metrics
            metrics["consistency_total"] += 1
            if is_consistent_correct:
                metrics["consistency_correct"] += 1

            # Update subject-level consistency metrics
            metrics["by_subject"][subject]["consistency_total"] += 1
            if is_consistent_correct:
                metrics["by_subject"][subject]["consistency_correct"] += 1

        # Save problem evaluation
        evaluation_results.append(problem_evaluation)

    # Process all problems
    for problem_idx, problem_data in enumerate(problems_data):
        if problem_idx < len(results):
            process_problem_samples(problem_idx, problem_data, results[problem_idx])

    # Calculate final metrics
    pass_at_1 = metrics["total"]["correct"] / metrics["total"]["total"] if metrics["total"]["total"] > 0 else 0
    cons_at_sample = (
        metrics["consistency_correct"] / metrics["consistency_total"]
        if metrics["consistency_total"] > 0
        else 0
    )

    subject_metrics = {}
    for subject, data in metrics["by_subject"].items():
        subject_pass_at_1 = (
            data["correct"] / data["total"] if data["total"] > 0 else 0
        )
        subject_cons_at_sample = (
            data["consistency_correct"] / data["consistency_total"]
            if data["consistency_total"] > 0
            else 0
        )
        avg_output_tokens = (
            np.mean(data["output_tokens"]) if data["output_tokens"] else 0
        )
        avg_response_length = (
            np.mean(data["response_lengths"]) if data["response_lengths"] else 0
        )

        avg_correct_tokens = (
            np.mean(data["correct_output_tokens"])
            if data["correct_output_tokens"]
            else 0
        )
        avg_correct_length = (
            np.mean(data["correct_response_lengths"])
            if data["correct_response_lengths"]
            else 0
        )

        avg_incorrect_tokens = (
            np.mean(data["incorrect_output_tokens"])
            if data["incorrect_output_tokens"]
            else 0
        )
        avg_incorrect_length = (
            np.mean(data["incorrect_response_lengths"])
            if data["incorrect_response_lengths"]
            else 0
        )

        subject_metrics[subject] = {
            "pass@1": subject_pass_at_1,
            "cons@SAMPLE": subject_cons_at_sample,
            "total_samples": data["total"],
            "correct_samples": data["correct"],
            "avg_output_tokens": avg_output_tokens,
            "avg_response_length": avg_response_length,
            "avg_correct_tokens": avg_correct_tokens,
            "avg_correct_length": avg_correct_length,
            "avg_incorrect_tokens": avg_incorrect_tokens,
            "avg_incorrect_length": avg_incorrect_length,
            "consistency_correct": data["consistency_correct"],
            "consistency_total": data["consistency_total"],
        }

    total_avg_output_tokens = (
        np.mean(metrics["total"]["output_tokens"])
        if metrics["total"]["output_tokens"]
        else 0
    )
    total_avg_response_length = (
        np.mean(metrics["total"]["response_lengths"])
        if metrics["total"]["response_lengths"]
        else 0
    )

    correct_avg_tokens = (
        np.mean(metrics["total"]["correct_output_tokens"])
        if metrics["total"]["correct_output_tokens"]
        else 0
    )
    correct_avg_length = (
        np.mean(metrics["total"]["correct_response_lengths"])
        if metrics["total"]["correct_response_lengths"]
        else 0
    )

    incorrect_avg_tokens = (
        np.mean(metrics["total"]["incorrect_output_tokens"])
        if metrics["total"]["incorrect_output_tokens"]
        else 0
    )
    incorrect_avg_length = (
        np.mean(metrics["total"]["incorrect_response_lengths"])
        if metrics["total"]["incorrect_response_lengths"]
        else 0
    )

    final_metrics = {
        "pass@1": pass_at_1,
        "cons@SAMPLE": cons_at_sample,
        "total_samples": metrics["total"]["total"],
        "correct_samples": metrics["total"]["correct"],
        "consistency_correct": metrics["consistency_correct"],
        "consistency_total": metrics["consistency_total"],
        "avg_output_tokens": total_avg_output_tokens,
        "avg_response_length": total_avg_response_length,
        "avg_correct_tokens": correct_avg_tokens,
        "avg_correct_length": correct_avg_length,
        "avg_incorrect_tokens": incorrect_avg_tokens,
        "avg_incorrect_length": incorrect_avg_length,
        "by_subject": subject_metrics,
    }

    # Print metrics
    language = "Chinese" if is_chinese else "English"
    print(f"\n===== {language} EVALUATION RESULTS =====")
    print(f"pass@1: {final_metrics['pass@1']:.4f}")
    print(f"cons@{sample_size}: {final_metrics['cons@SAMPLE']:.4f}")
    print(f"Avg output tokens: {final_metrics['avg_output_tokens']:.2f}")
    print(f"Avg correct output tokens: {final_metrics['avg_correct_tokens']:.2f}")
    print(f"Avg incorrect output tokens: {final_metrics['avg_incorrect_tokens']:.2f}")

    print("\nResults by subject:")
    for subject, metrics_data in final_metrics["by_subject"].items():
        print(f"  {subject}:")
        print(f"    pass@1: {metrics_data['pass@1']:.4f}")
        print(f"    cons@{sample_size}: {metrics_data['cons@SAMPLE']:.4f}")
        print(f"    Avg output tokens: {metrics_data['avg_output_tokens']:.2f}")
        print(f"    Avg correct output tokens: {metrics_data['avg_correct_tokens']:.2f}")
        print(f"    Avg incorrect output tokens: {metrics_data['avg_incorrect_tokens']:.2f}")
        print(f"    Total samples: {metrics_data['total_samples']}")

    # Save metrics to log directory
    with open(f"{log_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model,
                "sample_count": sample_size,
                "problem_count": problem_count,
                "language": language,
                "metrics": final_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nResults saved to {log_dir}")

    return evaluation_results


if __name__ == "__main__":
    try:
        args = parse_arguments()
        run_evaluation(args)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
