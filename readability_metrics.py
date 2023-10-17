import ast
from itertools import combinations




def compute_textual_coherence(code_snippet):
    # Parse the source code and build the Abstract Syntax Tree (AST)
    tree = ast.parse(code_snippet)

    # Extract all syntactic blocks (All branching statements)
    syntactic_blocks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) or isinstance(node, ast.For) or isinstance(node, ast.While) or isinstance(node, ast.With) or isinstance(node, ast.Try) or isinstance(node, ast.ExceptHandler) or isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef) or isinstance(node, ast.AsyncFunctionDef) or isinstance(node, ast.AsyncFor) or isinstance(node, ast.AsyncWith):
            syntactic_blocks.append(node)
        if isinstance(node, ast.If):
            if node.orelse:
                syntactic_blocks.append(node.orelse)
        
    import pdb; pdb.set_trace()
    # Compute vocabulary overlap for all pairs of distinct syntactic blocks
    vocab_overlap = []
    for pair in combinations(syntactic_blocks, 2):
        # Extract the vocabulary of each syntactic block
        vocab1 = set(ast.dump(pair[0]).split())
        vocab2 = set(ast.dump(pair[1]).split())

        # Compute the vocabulary overlap
        vocab_overlap.append(len(vocab1.intersection(vocab2)) / len(vocab1.union(vocab2)))

    import pdb ; pdb.set_trace()
    # Compute textual coherence as the max, min, or average overlap
    tc_max = max(vocab_overlap)
    tc_min = min(vocab_overlap)
    tc_avg = sum(vocab_overlap) / len(vocab_overlap)

    return tc_max, tc_min, tc_avg

# Example usage with the provided code snippet in python having if conditions
code_snippet = """
if a > 0:
    print("a is positive")
elif a < 0:
    print("a is negative")
else:
    print("a is zero")
"""

result = compute_textual_coherence(code_snippet)
print("Textual coherence (TC) - Max: {}, Min: {}, Average: {}".format(*result))
