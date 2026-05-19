import json
import re
import argparse
import sys
from config import *


def split_by_regex(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list:
    """
    Разделение корейского текста на предложения с помощью regex.
    Работает без дополнительных зависимостей.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    segments = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) <= max_chars:
            segments.append(paragraph)
            continue

        # Корейские предложения обычно заканчиваются на: 다. 요. 까? 죠. и т.д.
        sentences = re.split(
            r'(?<=[.!?。！？])\s+|(?<=[다요죠음임니까][.!?])\s*|(?<=[다요죠음임니까])\s+(?=[A-Z가-힣"\'\(「『])',
            paragraph
        )

        if len(sentences) <= 1:
            sentences = re.split(r'(?<=[.!?。！？])\s*', paragraph)

        current_segment = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_segment) + len(sentence) + 1 <= max_chars:
                current_segment = (current_segment + " " + sentence).strip()
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = sentence

        if current_segment:
            segments.append(current_segment)

    return segments


def split_by_kss(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list:
    """
    Разделение с помощью kss (Korean Sentence Splitter).
    Требует: pip install kss pecab
    """
    try:
        import kss
    except ImportError:
        print("ОШИБКА: kss не установлен. Установите: pip install kss pecab")
        print("Или используйте режим regex: python stage1_split.py --method regex")
        sys.exit(1)

    paragraphs = re.split(r'\n\s*\n', text)
    segments = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        try:
            sentences = kss.split_sentences(paragraph, backend='pecab')
        except Exception:
            try:
                sentences = kss.split_sentences(paragraph)
            except Exception:
                sentences = [paragraph]

        current_segment = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_segment) + len(sentence) + 1 <= max_chars:
                current_segment = (current_segment + " " + sentence).strip()
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = sentence

        if current_segment:
            segments.append(current_segment)

    return segments


def main():
    parser = argparse.ArgumentParser(description='Этап 1: Разделение текста на сегменты')
    parser.add_argument('--method', choices=['kss', 'regex'],
                        default='regex',
                        help='Метод разделения (default: regex)')
    parser.add_argument('--max-chars', type=int, default=MAX_SEGMENT_CHARS,
                        help=f'Максимальный размер сегмента (default: {MAX_SEGMENT_CHARS})')
    parser.add_argument('--input', type=str, default=SOURCE_FILE)
    parser.add_argument('--output', type=str, default=SEGMENTS_FILE)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ОШИБКА: Файл не найден: {args.input}")
        print(f"Создайте файл {args.input} с корейским текстом для перевода.")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Исходный текст: {len(text)} символов")
    print(f"Метод разделения: {args.method}")
    print(f"Макс. размер сегмента: {args.max_chars} символов")
    print("-" * 50)

    if args.method == 'kss':
        segments = split_by_kss(text, args.max_chars)
    else:
        segments = split_by_regex(text, args.max_chars)

    # Объединяем слишком короткие сегменты с предыдущим
    filtered_segments = []
    for seg in segments:
        if filtered_segments and len(seg) < MIN_SEGMENT_CHARS:
            filtered_segments[-1] += "\n" + seg
        else:
            filtered_segments.append(seg)

    output_data = {
        "metadata": {
            "source_file": args.input,
            "method": args.method,
            "max_chars": args.max_chars,
            "total_segments": len(filtered_segments),
            "total_chars": sum(len(s) for s in filtered_segments)
        },
        "segments": [
            {"id": i + 1, "text": seg, "char_count": len(seg)}
            for i, seg in enumerate(filtered_segments)
        ]
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Результат: {len(filtered_segments)} сегментов")
    print(f"Сохранено в: {args.output}")
    print("\nПревью первых 5 сегментов:")
    print("=" * 50)
    for i, seg in enumerate(filtered_segments[:5]):
        preview = seg[:100] + "..." if len(seg) > 100 else seg
        print(f"  [{i+1}] ({len(seg)} симв.) {preview}")
    if len(filtered_segments) > 5:
        print(f"  ... и ещё {len(filtered_segments) - 5} сегментов")


if __name__ == "__main__":
    main()
