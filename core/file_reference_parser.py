from pathlib import Path
from typing import List

class FileReferenceParser:
    @staticmethod
    def parse_message(message: str) -> str:
        """
        Parse a message containing file references (prefixed with '@')
        and return the message with references replaced with the
        annotated file contents.

        Args:
            message: Input message potentially containing file references

        Returns:
            str: The message with file contents inserted into it

        Raises:
            ValueError: If line numbers are invalid or file doesn't exist
        """
        parts = []
        original_parts = message.split()

        for i, part in enumerate(original_parts):
            if part.startswith("@"):
                try:
                    start_line = 0
                    end_line = -1
                    filename = part[1:]

                    # Handle line numbers if present
                    if ":" in filename:
                        filename, line_numbers = filename.split(":", 1)

                        if "-" in line_numbers:
                            start_line, end_line = [int(x) for x in line_numbers.split("-", 1)]
                            start_line -= 1
                        else:
                            start_line = int(line_numbers) - 1
                            end_line = start_line + 1

                    path = Path(filename).resolve()
                    if not path.exists():
                        raise FileNotFoundError(f"File not found: {path}")

                    contents = FileReferenceParser.read_file_contents(path).splitlines()

                    # Validate line numbers
                    if start_line < 0 or end_line > len(contents):
                        raise ValueError(
                            f"Invalid line range for '{filename}': "
                            f"{start_line + 1}-{end_line if end_line != -1 else 'EOF'}"
                        )

                    # Determine section description
                    if end_line == -1:
                        section = "full file"
                    elif end_line > start_line + 1:
                        section = f"lines {start_line + 1} to {end_line}"
                    else:
                        section = f"line {start_line + 1}"

                    # Add file section header
                    parts.append(f"\n=== START FILE SECTION '{filename}', {section} ===\n")

                    # Add content based on line range
                    if end_line == -1:
                        parts.append("\n".join(contents))
                    elif end_line > start_line + 1:
                        parts.append("\n".join(contents[start_line:end_line]))
                    else:
                        parts.append(contents[start_line])

                    parts.append(f"\n=== END FILE SECTION '{filename}' ===\n")

                except (ValueError, FileNotFoundError) as e:
                    # Keep original reference if there's an error
                    parts.append(part)
            else:
                parts.append(part)

        return " ".join(parts)

    @staticmethod
    def read_file_contents(file_path: Path) -> str:
        """Read contents of a file with proper encoding handling.

        Args:
            file_path: Path to the file to read

        Returns:
            str: File contents as string

        Raises:
            UnicodeDecodeError: If file cannot be decoded
        """
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1 for binary files
            return file_path.read_text(encoding='latin-1')

def main():
    test_cases = [
        "Read the contents of @core/file_reference_parser.py please",
        "Check lines 5-10 of @core/file_reference_parser.py:5-10",
        "Look at line 20 of @nonexistent_file.txt:20",
        "Multiple @file1.py and @file2.py references"
    ]

    for message in test_cases:
        print(f"\nOriginal: {message}")
        print("Parsed:")
        print(FileReferenceParser.parse_message(message))
        print("-" * 50)

if __name__ == "__main__":
    main()
