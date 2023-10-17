import ast
from pathlib import Path

def get_ast(code: str):
    print(ast.dump(ast.parse(code), indent=2))

def main():
    code_file = Path("test.py").read_text()
    get_ast(code_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
