import os
import fnmatch

# Patterns for directories and files to ignore.
# These are typically basenames or fnmatch patterns for basenames (e.g., '*.log').
# Patterns targeting directories should not end with '/' (e.g., use 'build', not 'build/').
DEFAULT_IGNORE_PATTERNS = [
    # Version control
    '.git', '.svn', '.hg',
    # IDE/Editor specific
    '.vscode', '.idea', '.vs', '.project', '.classpath', '.settings', '.tmPreferences',
    # Python specific
    '__pycache__', '*.pyc', '*.pyo', '.pytest_cache', '.mypy_cache', '*.egg-info', 'pip-wheel-metadata',
    'venv', 'env', '.venv', '.env', 'pip-selfcheck.json',
    'build', 'dist', 'sdist', 'wheels', '*.manifest', 'MANIFEST',
    # Node.js
    'node_modules', '*.log', # Generic log files (often verbose and not primary structure)
    'npm-debug.log*', 'yarn-debug.log*', 'yarn-error.log*',
    # Java/Maven/Gradle
    'target', 'bin', 'out', '.gradle', # .gradle FOLDER (dot-gradle files like 'build.gradle' are scripts)
    # C/C++
    '*.o', '*.obj', '*.so', '*.dll', '*.dylib', '*.exe', '*.out', '*.a', '*.lib',
    # OS specific
    '.DS_Store', 'Thumbs.db', 'desktop.ini',
    # General cache/temp folders or files
    '.cache', 'cache',
    'temp', 'tmp', '.temp', '.tmp',
    '*~', '*.swp', '*.swo', '*.bak', # Editor backup/swap files
    '*.tmp', # General temporary files
    # Streamlit (from original requirement)
    '.streamlit',
    # LaTeX auxiliary files
    '*.aux', '*.lof', '*.lot', '*.fls', '*.fdb_latexmk', '*.synctex.gz', '_minted*', '*.toc',
]

# --- File Type Categorizations ---
# Exact filenames (case-insensitive) to be treated as script/code
SCRIPT_FILENAMES_EXACT = {
    'makefile', 'gnumakefile', 'rakefile', 'gemfile', 'procfile', 'dockerfile', 'vagrantfile',
    'readme', 'license', 'contributing', 'authors', 'changelog', 'news', 'copying', # Common project meta files
    'build.xml', 'pom.xml', 'setup.py', 'requirements.txt', 'constraints.txt', 'pipfile', 'poetry.lock',
    'package.json', 'package-lock.json', 'yarn.lock', # Essential project structure files
    'tsconfig.json', 'angular.json', 'webpack.config.js', 'babel.config.js', 'jest.config.js', 'vite.config.js',
    '.prettierrc', '.eslintrc.js', '.eslintrc.json', '.editorconfig', '.gitattributes', '.gitmodules', '.env.example',
    'pyproject.toml', # Often contains build system defs, might also be data if only project metadata
}

# File extensions (lowercase, starting with '.') for script/code files
SCRIPT_EXTENSIONS = {
    # Python
    '.py', '.pyw', '.ipynb',
    # Shell / Batch
    '.sh', '.bash', '.zsh', '.csh', '.ksh', '.fish', '.bat', '.cmd', '.ps1',
    # Web
    '.js', '.jsx', '.ts', '.tsx', '.html', '.htm', '.css', '.scss', '.less', '.sass', '.vue', '.svelte', '.php',
    # C-family
    '.c', '.h', '.cpp', '.hpp', '.cs', '.m', '.mm', # C, C++, C#, Obj-C
    # Java, Kotlin, Scala, Groovy
    '.java', '.kt', '.kts', '.scala', '.groovy', '.gradle',
    # Go, Rust, Dart, Zig
    '.go', '.rs', '.dart', '.zig',
    # Ruby, Perl, Lua
    '.rb', '.pl', '.pm', '.t', '.lua',
    # Swift
    '.swift',
    # SQL
    '.sql', '.ddl', '.dml',
    # Config/Build/Structure files (text-based, define behavior or structure)
    '.tf', '.tfvars', '.hcl', '.jsonnet', '.libsonnet',
    # Project files (if not large data and define structure)
    '.csproj', '.vbproj', '.fsproj', '.vcproj', '.vcxproj', '.sln',
    # VB
    '.vb', '.vbs',
    # Documentation as code
    '.md', '.rst', '.tex', '.adoc', '.asciidoc',
    # YAML/YML - often config, scripts, or structured human-readable data. Treat as script-like.
    '.yaml', '.yml',
    # Other languages
    '.ex', '.exs', # Elixir
    '.elm', # Elm
    '.erl', '.hrl', # Erlang
    '.clj', '.cljs', '.cljc', # Clojure
    '.f', '.f90', '.for', # Fortran
    '.pas', '.pp', # Pascal
    '.asm', '.s', # Assembly
    '.d', # D language
    '.hs', '.lhs', # Haskell
    '.R', '.r', # R scripts
    '.applescript', '.scpt', # AppleScript
    '.pde', # Processing
    # Common config file formats (often driving behavior)
    '.conf', '.config', '.cfg', '.ini', '.properties',
    # Systemd & Desktop files
    '.service', '.socket', '.target', '.mount', '.automount', '.timer', '.path',
    '.desktop',
    # Other structured text files defining behavior/rules
    '.rules', # e.g., udev rules
    '.sls', # SaltStack state files
    '.rego', # Open Policy Agent
    '.spec', # RPM Spec files
    '.proto', # Protocol Buffers
}

# File extensions (lowercase, starting with '.') for data files (subject to 5-item limit per type)
DATA_EXTENSIONS = {
    # Common data interchange formats
    '.json', '.xml', '.toml', # Note: some specific .json/.toml are in SCRIPT_FILENAMES_EXACT
    '.csv', '.tsv', '.parquet', '.avro', '.orc', '.feather',
    # Audio
    '.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a', '.opus',
    # Video
    '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv',
    # Images
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.svg', '.webp', '.ico', '.heic', '.heif', '.psd', '.ai',
    # Documents (that are not typically "code" or "config structure")
    '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.odt', '.ods', '.odp', '.rtf',
    # Text / Generic data (specific log files are usually ignored by DEFAULT_IGNORE_PATTERNS)
    '.txt', '.dat', '.data', '.log', # General logs, if not ignored earlier
    # Binary data / Serialized objects
    '.bin', '.pickle', '.pkl', '.joblib', '.arrow',
    # ML models/data
    '.pt', '.pth', '.h5', '.hdf5', '.pb', '.onnx', '.tflite', '.ckpt', '.model', '.weights', '.safetensors',
    # Databases / Database exports
    '.db', '.sqlite', '.sqlite3', '.mdb', '.accdb', '.dump', '.sqlitedb',
    # Archives (can be numerous if not code libraries)
    '.zip', '.tar', '.gz', '.bz2', '.rar', '.7z', '.jar', '.war', '.ear',
    # NumPy arrays
    '.npy', '.npz',
    # Font files
    '.ttf', '.otf', '.woff', '.woff2', '.eot',
    # GIS / GPS data
    '.gpx', '.kml', '.kmz', '.shp', '.dbf', '.shx', '.geojson',
    # Bioinformatics
    '.fasta', '.fastq', '.pdb', '.gb', '.genbank', '.sam', '.bam',
    # Other potentially numerous data files
    '.mat', # Matlab data files
    '.RData', '.rds', # R data files
    # CAD files
    '.dwg', '.dxf', '.stl', '.obj', # .obj can be 3D model data or C object file (latter ignored by DEFAULT_IGNORE_PATTERNS)
}

# --- Helper Functions ---

memo_has_code = {} # Memoization cache for has_code_recursively

def matches_any_pattern(path_to_check, patterns):
    """
    Checks if path_to_check (relative to project root, or basename) matches any pattern.
    """
    normalized_path = path_to_check.replace(os.sep, '/')
    # Use basename for patterns that don't contain '/', otherwise use full normalized path
    name_of_item = os.path.basename(normalized_path)
    if not name_of_item and normalized_path.endswith('/'): # Handle 'somedir/' -> 'somedir'
        name_of_item = os.path.basename(normalized_path[:-1])
    elif not name_of_item and normalized_path: # Handle cases like '/' if path_to_check was '/'
        name_of_item = normalized_path


    for pat in patterns:
        if '/' in pat: # Pattern contains path separators, match against full path
            if fnmatch.fnmatch(normalized_path, pat):
                return True
        else: # Pattern is a basename or glob for basename (e.g. '*.pyc', 'node_modules')
            if fnmatch.fnmatch(name_of_item, pat):
                return True
    return False

def has_code_recursively(dir_abs_path, project_abs_start_path, all_ignore_patterns):
    """
    Checks if a directory or any of its subdirectories contain script/code files.
    Respects ignore patterns. Uses memoization.
    `dir_abs_path` is the absolute path to the directory to check.
    `project_abs_start_path` is the absolute path to the root of the project scan.
    """
    if dir_abs_path in memo_has_code:
        return memo_has_code[dir_abs_path]

    # Path for matching patterns should be relative to project_abs_start_path or just basename.
    # This covers if dir_abs_path itself is on the ignore list (e.g., 'node_modules').
    dir_rel_path_to_project_start = os.path.relpath(dir_abs_path, project_abs_start_path)
    # If dir_abs_path is project_abs_start_path, relpath is '.', use basename for matching.
    path_for_ignore_check_self = dir_rel_path_to_project_start if dir_rel_path_to_project_start != '.' else os.path.basename(dir_abs_path)

    if matches_any_pattern(path_for_ignore_check_self, all_ignore_patterns):
        memo_has_code[dir_abs_path] = False
        return False

    for current_root_abs, sub_dirs_local, sub_files_local in os.walk(dir_abs_path, topdown=True):
        # Path of current_root_abs relative to the overall project start path
        current_root_rel_to_project_start = os.path.relpath(current_root_abs, project_abs_start_path)
        if current_root_rel_to_project_start == '.': current_root_rel_to_project_start = "" # For items directly in project_abs_start_path

        # Filter sub_dirs_local for this recursive walk based on ignore patterns
        # Paths for matching subdirectories are relative to project_abs_start_path
        sub_dirs_local[:] = [
            sd for sd in sub_dirs_local
            if not matches_any_pattern(os.path.join(current_root_rel_to_project_start, sd), all_ignore_patterns)
        ]
        sub_dirs_local.sort() # Ensure consistent traversal order

        for f_name in sub_files_local:
            # Path of file relative to the overall project start path
            file_rel_to_project_start = os.path.join(current_root_rel_to_project_start, f_name)
            if matches_any_pattern(file_rel_to_project_start, all_ignore_patterns):
                continue

            f_name_lower = f_name.lower()
            _, ext_lower = os.path.splitext(f_name_lower)
            if f_name_lower in SCRIPT_FILENAMES_EXACT or ext_lower in SCRIPT_EXTENSIONS:
                memo_has_code[dir_abs_path] = True
                return True

    memo_has_code[dir_abs_path] = False
    return False

# --- Main Function ---
def print_directory_structure(start_path='.'):
    abs_start_path = os.path.abspath(start_path)

    global memo_has_code
    memo_has_code = {} # Clear memoization cache for each run

    all_ignore_patterns = DEFAULT_IGNORE_PATTERNS

    for root_abs, dirs_from_walk, files_from_walk in os.walk(abs_start_path, topdown=True):
        # Path of root_abs relative to the initial start_path for display and consistent logic
        rel_root_to_start = os.path.relpath(root_abs, abs_start_path)
        if rel_root_to_start == '.': rel_root_to_start = ""

        # --- 1. Filter Dirs and Files at current level based on global ignore patterns ---
        # These are names within root_abs. Build their path relative to abs_start_path for matching.

        # Filter and sort subdirectories for current level processing and for pruning walk
        # `d_name` is just a name; `os.path.join(rel_root_to_start, d_name)` is its path from project root.
        processed_dirs_current_level = []
        for d_name in sorted(dirs_from_walk): # Sort for consistent "first 8" logic and display order
            path_for_d_ignore_check = os.path.join(rel_root_to_start, d_name)
            if not matches_any_pattern(path_for_d_ignore_check, all_ignore_patterns):
                processed_dirs_current_level.append(d_name)

        # Filter and sort files for current level display
        # `f_name` is just a name; `os.path.join(rel_root_to_start, f_name)` is its path from project root.
        processed_files_current_level = []
        for f_name in sorted(files_from_walk): # Sort for consistent display order
            path_for_f_ignore_check = os.path.join(rel_root_to_start, f_name)
            if not matches_any_pattern(path_for_f_ignore_check, all_ignore_patterns):
                processed_files_current_level.append(f_name)

        # --- 2. Print Current Directory Name ---
        level = rel_root_to_start.count(os.sep) if rel_root_to_start else 0
        indent = ' ' * 4 * level

        current_dir_name_display = os.path.basename(root_abs)
        if not current_dir_name_display and root_abs == os.path.sep: # Handle root '/' on Unix
            current_dir_name_display = os.path.sep
        elif not current_dir_name_display and len(root_abs) == 2 and root_abs[1] == ':': # Handle 'C:' on Windows
             current_dir_name_display = root_abs
        elif not current_dir_name_display : # Fallback for other empty basename cases
             current_dir_name_display = root_abs if root_abs else "."


        print(f"{indent}{current_dir_name_display}/")
        sub_indent = ' ' * 4 * (level + 1)

        # --- 3. Print Files in Current Directory (Categorized) ---
        script_code_files = []
        data_files_grouped = {} # Key: extension (lowercase), Value: list of filenames
        other_display_files = []

        for f_name in processed_files_current_level: # Already sorted and filtered
            f_name_lower = f_name.lower()
            _, ext_lower = os.path.splitext(f_name_lower)

            is_script = False
            if f_name_lower in SCRIPT_FILENAMES_EXACT or ext_lower in SCRIPT_EXTENSIONS:
                is_script = True

            if is_script:
                script_code_files.append(f_name)
            elif ext_lower in DATA_EXTENSIONS:
                data_files_grouped.setdefault(ext_lower, []).append(f_name)
            else:
                other_display_files.append(f_name)

        # Print categorized files (order: Scripts, Data, Others)
        for f_name in script_code_files: print(f"{sub_indent}{f_name}")

        for ext_key in sorted(data_files_grouped.keys()): # Sort data file groups by extension
            file_list = data_files_grouped[ext_key] # Files within group are already sorted
            if len(file_list) > 5:
                for i in range(5): print(f"{sub_indent}{file_list[i]}")
                print(f"{sub_indent}... and {len(file_list) - 5} more {ext_key} files")
            else:
                for f_item in file_list: print(f"{sub_indent}{f_item}")

        for f_name in other_display_files: print(f"{sub_indent}{f_name}")

        # --- 4. Prune Directories for os.walk's next iteration (modifying dirs_from_walk[:]) ---
        # `processed_dirs_current_level` contains subdirectories of `root_abs` that are:
        #   a) Not globally ignored.
        #   b) Sorted alphabetically.
        dirs_to_walk_next = []
        data_only_dirs_kept_count = 0
        data_only_dirs_skipped_count = 0

        apply_dir_count_limit = len(processed_dirs_current_level) > 8

        for d_name in processed_dirs_current_level: # Iterating in sorted order
            # Absolute path of the subdirectory being considered for recursive walk
            dir_abs_path_to_check = os.path.join(root_abs, d_name)

            contains_code = has_code_recursively(dir_abs_path_to_check, abs_start_path, all_ignore_patterns)

            if contains_code:
                dirs_to_walk_next.append(d_name)
            else: # This subdirectory is data-only (recursively)
                if not apply_dir_count_limit or data_only_dirs_kept_count < 8:
                    dirs_to_walk_next.append(d_name)
                    data_only_dirs_kept_count += 1
                else:
                    data_only_dirs_skipped_count += 1

        dirs_from_walk[:] = dirs_to_walk_next # This prunes os.walk. Order is preserved.

        if data_only_dirs_skipped_count > 0:
            print(f"{sub_indent}... and {data_only_dirs_skipped_count} more data-only sub-directories not shown.")


if __name__ == "__main__":
    # You can change the starting path here if needed:
    # print_directory_structure("/path/to/your/project")
    print_directory_structure(".")