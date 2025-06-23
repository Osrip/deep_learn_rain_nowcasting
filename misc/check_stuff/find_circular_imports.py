#!/usr/bin/env python3
"""
Detect circular import patterns in the codebase.
Finds intra-package imports that could cause circular dependencies.
"""

import os
import re
from pathlib import Path


def find_intra_package_imports():
    """Find modules that import from their own package level."""

    packages = ['helper', 'model', 'evaluation', 'data_pre_processing', 'training_utils', 'plotting', 'tests']
    circular_patterns = []

    for package in packages:
        if not os.path.exists(package):
            continue

        print(f"\nChecking {package}/...")

        for file_path in Path(package).glob('*.py'):
            if file_path.name == '__init__.py':
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                continue

            # Look for same-package imports
            pattern = rf'from {package} import'
            matches = re.findall(pattern, content)

            if matches:
                circular_patterns.append({
                    'file': str(file_path),
                    'package': package,
                    'problem': f'File in {package}/ imports from {package}',
                    'matches': len(matches)
                })

                print(f"  {file_path.name} imports from {package}")

                # Show the problematic lines
                for line_num, line in enumerate(content.split('\n'), 1):
                    if pattern in line:
                        print(f"    Line {line_num}: {line.strip()}")

    return circular_patterns


def suggest_fixes(circular_patterns):
    """Print suggested fixes for circular imports."""

    print(f"\n{'-' * 50}")
    print("SUGGESTED FIXES:")
    print(f"{'-' * 50}")

    for pattern in circular_patterns:
        file_path = pattern['file']
        package = pattern['package']

        print(f"\n{file_path}")
        print(f"  Problem: {pattern['problem']}")
        print(f"  Fix: Change 'from {package} import X' to 'from .module_name import X'")
        print(f"  Example: from {package} import func -> from .some_module import func")


def main():
    print("Scanning for circular import patterns...")
    print("-" * 50)

    circular_patterns = find_intra_package_imports()

    if circular_patterns:
        print(f"\nFound {len(circular_patterns)} potential circular import issues")
        suggest_fixes(circular_patterns)

        print(f"\n{'-' * 50}")
        print("SUMMARY:")
        print("Replace 'from package import X' with 'from .module import X'")
        print("when importing within the same package directory.")
        print(f"{'-' * 50}")

    else:
        print("\nNo circular import patterns detected")
        print("Package imports look clean.")


if __name__ == "__main__":
    main()