import json
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            obj = json.loads(line)
            print("UID:", obj["uid"])
            print("GOLD:", obj["gold"])
            for cp in obj["checkpoints"]:
                print(f"\n--- t={cp['t']} {cp['mode']} ---")
                print("answer:", cp.get("answer") or cp.get("verified_answer"))
                print("conf:", cp.get("conf"))
                print("correct:", cp["correct"])
            print("\nTTC:", obj["labels"]["ttc"])
            print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
