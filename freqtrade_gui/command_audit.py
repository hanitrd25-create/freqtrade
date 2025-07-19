import os
import ast

def get_cli_commands():
    """
    Parses the freqtrade/commands directory to extract CLI commands.
    """
    commands = []
    commands_dir = "freqtrade/commands"
    for filename in os.listdir(commands_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            filepath = os.path.join(commands_dir, filename)
            with open(filepath, "r") as f:
                try:
                    tree = ast.parse(f.read(), filename=filename)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # This is a simple heuristic. A more robust solution would be
                            # to look for how these functions are registered with the CLI parser.
                            if node.name.startswith("start_") or "args" in [a.arg for a in node.args.args]:
                                commands.append({
                                    "file": filename,
                                    "command": node.name,
                                    "docstring": ast.get_docstring(node)
                                })
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")
    return commands

def get_rpc_commands():
    """
    Parses the freqtrade/rpc directory to extract RPC commands.
    """
    commands = []
    rpc_dir = "freqtrade/rpc"
    for filename in os.listdir(rpc_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            filepath = os.path.join(rpc_dir, filename)
            with open(filepath, "r") as f:
                try:
                    tree = ast.parse(f.read(), filename=filename)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # This is a simple heuristic. A more robust solution would be
                            # to look for how these functions are registered with the RPC handler.
                            if node.name.startswith("_") and "update" in [a.arg for a in node.args.args] and "context" in [a.arg for a in node.args.args]:
                                commands.append({
                                    "file": filename,
                                    "command": node.name,
                                    "docstring": ast.get_docstring(node)
                                })
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")

    return commands

if __name__ == "__main__":
    cli_commands = get_cli_commands()
    rpc_commands = get_rpc_commands()

    print("CLI Commands:")
    for cmd in cli_commands:
        print(f"  - File: {cmd['file']}, Command: {cmd['command']}")

    print("\nRPC Commands:")
    for cmd in rpc_commands:
        print(f"  - File: {cmd['file']}, Command: {cmd['command']}")
