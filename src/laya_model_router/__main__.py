"""Manual testing: uv run python -m laya_model_router "<task text>" """

import argparse
import json

from .router import ModelRouter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a task's difficulty with Laya and pick a model for it."
    )
    parser.add_argument("text", help="The task description to classify.")
    args = parser.parse_args()

    router = ModelRouter()
    result = router.pick_model(args.text)
    print(
        json.dumps(
            {
                "tier": result.tier,
                "model": result.model,
                "score": result.score,
                "confidence": result.confidence,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
