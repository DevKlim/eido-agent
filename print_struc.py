import os
import fnmatch

DEFAULT_IGNORE_LIST = ['.git', 'node_modules', 'venv', '__pycache__', '.vscode', '.idea', '.DS_Store', '*.pyc', '*.log', '.streamlit']

def load_gitignore_patterns(gitignore_path):
    patterns = []
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Handle directory patterns (ending with /) for fnmatch
                # fnmatch treats 'dir/' and 'dir' differently if not careful
                # For simplicity, we'll let fnmatch handle it, but could normalize by removing trailing /
                # if line.endswith('/'):
                #     line = line[:-1]
                patterns.append(line)
    return patterns

def matches_any_pattern(path_to_check, patterns, is_dir=False):
    """
    Returns True if path_to_check matches any pattern.
    Handles directory patterns (ending with /) by checking both with and without trailing slash.
    """
    # Normalize path for matching (e.g., use forward slashes)
    normalized_path = path_to_check.replace(os.sep, '/')

    for pat in patterns:
        # For directory patterns like 'some_dir/', match only if path_to_check is a directory
        if pat.endswith('/'):
            if is_dir and fnmatch.fnmatch(normalized_path + '/', pat): # Match 'some/dir/' against 'some_dir/'
                return True
            if is_dir and fnmatch.fnmatch(normalized_path, pat[:-1]): # Match 'some/dir' against 'some_dir' (if pattern was 'some_dir/')
                 return True
        else: # File pattern or general pattern
            if fnmatch.fnmatch(normalized_path, pat):
                return True
            # If it's a directory, also check if the pattern matches the directory name itself
            # e.g. pattern 'build' should match directory 'build'
            if is_dir and fnmatch.fnmatch(os.path.basename(normalized_path), pat):
                 return True

    return False


def print_directory_structure(start_path='.'):
    gitignore_path = os.path.join(start_path, '.gitignore')
    gitignore_patterns = load_gitignore_patterns(gitignore_path)
    
    # Combine hard-coded and .gitignore patterns
    # Ensure DEFAULT_IGNORE_LIST patterns are treated as general patterns
    # Gitignore patterns can be more complex (e.g. `foo/**/bar`)
    # For this script, we primarily use fnmatch which is simpler.
    
    # Create a combined list of patterns.
    # Add common default ignores not always in .gitignore for this script's purpose
    all_ignore_patterns = list(set(DEFAULT_IGNORE_LIST + gitignore_patterns))


    for root, dirs, files in os.walk(start_path, topdown=True):
        rel_root = os.path.relpath(root, start_path)
        if rel_root == '.': rel_root = "" # Top level

        # Filter directories before further descent
        original_dirs_count = len(dirs)
        dirs[:] = [d for d in dirs if not matches_any_pattern(os.path.join(rel_root, d), all_ignore_patterns, is_dir=True)]
        
        # Filter files
        files[:] = [f for f in files if not matches_any_pattern(os.path.join(rel_root, f), all_ignore_patterns, is_dir=False)]

        # Only print root if it's not ignored itself (important for start_path if it matches an ignore)
        if rel_root == "" or not matches_any_pattern(rel_root, all_ignore_patterns, is_dir=True):
            level = rel_root.count(os.sep) if rel_root else 0
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/") # Use os.path.basename for current dir name

            sub_indent = ' ' * 4 * (level + 1)
            for f in sorted(files): # Sort files for consistent output
                print(f"{sub_indent}{f}")
            # Also sort dirs for consistent printing order in next iteration of os.walk
            dirs.sort()


if __name__ == "__main__":
    print_directory_structure(".")