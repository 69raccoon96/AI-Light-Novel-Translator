import argparse
import json
import os

from config import *


def main():
    parser = argparse.ArgumentParser(
        description="Экспорт финального текста"
    )

    parser.add_argument(
        "--source",
        choices=["final", "translated"],
        default="final",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--format",
        choices=[
            "plain",
            "with_source",
            "side_by_side",
        ],
        default="plain",
    )

    args = parser.parse_args()

    if (
        args.source == "final"
        and os.path.exists(FINAL_FILE)
    ):
        with open(FINAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data["results"]
        text_key = "final_translation"
        fallback_key = "draft_translation"

    elif os.path.exists(TRANSLATED_FILE):
        with open(
            TRANSLATED_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        items = data["translations"]
        text_key = "translation"
        fallback_key = "translation"

    else:
        print("ОШИБКА: Нет файлов перевода!")
        return

    output_path = (
        args.output
        or os.path.join(
            OUTPUT_DIR,
            "final_text.txt",
        )
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for item in sorted(
            items,
            key=lambda x: x["id"],
        ):
            text = (
                item.get(text_key)
                or item.get(fallback_key, "")
            )

            if args.format == "plain":
                f.write(text + "\n\n")

            elif args.format == "with_source":
                f.write(
                    f"--- Сегмент "
                    f"{item['id']} ---\n"
                )

                f.write(
                    f"[КО] "
                    f"{item.get('source', '')}\n\n"
                )

                f.write(f"[РУ] {text}\n\n")

            elif args.format == "side_by_side":
                f.write("=" * 60 + "\n")
                f.write(f"#{item['id']}\n")

                f.write(
                    f"{item.get('source', '')}\n"
                )

                f.write("-" * 60 + "\n")
                f.write(f"{text}\n\n")

    print(
        f"Экспортировано "
        f"{len(items)} сегментов "
        f"→ {output_path}"
    )


if __name__ == "__main__":
    main()
