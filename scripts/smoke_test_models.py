import argparse
from src.models import load_llm, build_budget_prompt, parse_answer_and_conf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name (teachers or student)")
    ap.add_argument("--budget", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    llm = load_llm(args.model, load_in_4bit=args.load_in_4bit)

    problem = "If Sarah has 12 apples and gives 5 to Tom, how many apples does she have left?"
    prompt = build_budget_prompt(problem, budget_t=args.budget)

    out = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
    ans, conf = parse_answer_and_conf(out)

    print("=== RAW OUTPUT ===")
    print(out)
    print("\n=== PARSED ===")
    print("answer:", ans)
    print("conf:", conf)

if __name__ == "__main__":
    main()
