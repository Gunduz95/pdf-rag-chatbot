"""Command line:  python -m ragbot {extract|index|ask "質問"|demo}"""
import argparse

from . import answer, extract, indexer


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragbot", description="Ask questions about a PDF with local models.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract", help="PDF -> data/source.txt")
    sub.add_parser("index", help="chunk + embed + store in ChromaDB")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    sub.add_parser("demo", help="run the evaluation questions")
    args = parser.parse_args()

    if args.command == "extract":
        extract.extract_pdf()
    elif args.command == "index":
        indexer.build_index()
    elif args.command == "ask":
        answer.ask(args.question)
    else:
        for q in answer.DEMO_QUESTIONS:
            answer.ask(q)


if __name__ == "__main__":
    main()
