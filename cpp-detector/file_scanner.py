"""
C++ File Scanner Module
Recursively scans directories for C++ source files and organizes them by project.
"""

import os
from pathlib import Path
from typing import List, Dict, Set
from loguru import logger


class CppFileScanner:
    """Scanner for finding C++ files in directory structures."""

    # Common C++ file extensions
    CPP_EXTENSIONS = {'.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h', '.hh', '.hxx', '.h++'}

    # Directories to skip during scanning
    SKIP_DIRS = {
        '__pycache__', '.git', '.svn', '.hg', 'node_modules',
        'build', 'dist', '.venv', 'venv', 'env', '.tox',
        'CMakeFiles', '.cmake', 'Debug', 'Release'
    }

    def __init__(self, root_path: str):
        """
        Initialize the scanner with a root directory.

        Args:
            root_path: Root directory to scan for C++ files
        """
        self.root_path = Path(root_path).resolve()
        if not self.root_path.exists():
            raise ValueError(f"Root path does not exist: {root_path}")
        if not self.root_path.is_dir():
            raise ValueError(f"Root path is not a directory: {root_path}")

    def is_cpp_file(self, file_path: Path) -> bool:
        """
        Check if a file is a C++ source file.

        Args:
            file_path: Path to check

        Returns:
            True if file is a C++ source file
        """
        return file_path.suffix.lower() in self.CPP_EXTENSIONS

    def should_skip_dir(self, dir_path: Path) -> bool:
        """
        Check if a directory should be skipped during scanning.

        Args:
            dir_path: Directory to check

        Returns:
            True if directory should be skipped
        """
        return dir_path.name in self.SKIP_DIRS or dir_path.name.startswith('.')

    def scan_directory(self, min_size: int = 100, max_size: int = 1000000) -> List[Dict]:
        """
        Recursively scan directory for C++ files.

        Args:
            min_size: Minimum file size in bytes (default 100)
            max_size: Maximum file size in bytes (default 1MB)

        Returns:
            List of dictionaries containing file information
        """
        cpp_files = []

        logger.info(f"Scanning directory: {self.root_path}")

        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)

            # Remove directories to skip from the list (modifies in-place)
            dirs[:] = [d for d in dirs if not self.should_skip_dir(root_path / d)]

            for file in files:
                file_path = root_path / file

                if not self.is_cpp_file(file_path):
                    continue

                try:
                    file_size = file_path.stat().st_size

                    # Skip files that are too small or too large
                    if file_size < min_size or file_size > max_size:
                        logger.debug(f"Skipping {file_path}: size {file_size} bytes")
                        continue

                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # Calculate relative path from root
                    rel_path = file_path.relative_to(self.root_path)

                    cpp_files.append({
                        'file_path': str(file_path),
                        'relative_path': str(rel_path),
                        'file_name': file_path.name,
                        'extension': file_path.suffix,
                        'size': file_size,
                        'content': content,
                        'line_count': len(content.splitlines()),
                    })

                except (IOError, OSError) as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
                    continue

        logger.info(f"Found {len(cpp_files)} C++ files")
        return cpp_files

    def organize_by_project(self, files: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Organize files by their project directory.

        A project is identified by:
        1. Having a CMakeLists.txt or Makefile
        2. Being a direct subdirectory of the root
        3. Or fallback to parent directory grouping

        Args:
            files: List of file dictionaries from scan_directory

        Returns:
            Dictionary mapping project names to lists of files
        """
        projects = {}

        # First, try to identify projects by build files
        project_roots = self._find_project_roots()

        for file_info in files:
            file_path = Path(file_info['file_path'])

            # Find which project this file belongs to
            project_name = self._determine_project(file_path, project_roots)

            if project_name not in projects:
                projects[project_name] = []

            file_info['project'] = project_name
            projects[project_name].append(file_info)

        logger.info(f"Organized files into {len(projects)} projects")
        for project, files_list in projects.items():
            logger.info(f"  {project}: {len(files_list)} files")

        return projects

    def _find_project_roots(self) -> Dict[Path, str]:
        """
        Find project root directories by looking for build files.

        Returns:
            Dictionary mapping project root paths to project names
        """
        project_roots = {}
        build_files = {'CMakeLists.txt', 'Makefile', 'makefile', 'build.gradle',
                       'pom.xml', 'meson.build', 'configure.ac'}

        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)

            # Skip directories we should ignore
            dirs[:] = [d for d in dirs if not self.should_skip_dir(root_path / d)]

            # Check if this directory has any build files
            if any(bf in files for bf in build_files):
                project_name = root_path.name
                if root_path == self.root_path:
                    project_name = "root"
                project_roots[root_path] = project_name
                logger.debug(f"Found project: {project_name} at {root_path}")

        return project_roots

    def _determine_project(self, file_path: Path, project_roots: Dict[Path, str]) -> str:
        """
        Determine which project a file belongs to.

        Args:
            file_path: Path to the file
            project_roots: Dictionary of known project roots

        Returns:
            Project name
        """
        # Check if file is in any project root
        for root, name in project_roots.items():
            if file_path.is_relative_to(root):
                return name

        # Fallback: use the immediate subdirectory of root
        try:
            rel_path = file_path.relative_to(self.root_path)
            parts = rel_path.parts
            if len(parts) > 1:
                return parts[0]
            else:
                return "root"
        except ValueError:
            return "unknown"


def scan_cpp_files(root_path: str, organize: bool = True,
                   min_size: int = 100, max_size: int = 1000000) -> Dict:
    """
    Convenience function to scan for C++ files.

    Args:
        root_path: Root directory to scan
        organize: Whether to organize by project (default True)
        min_size: Minimum file size in bytes
        max_size: Maximum file size in bytes

    Returns:
        Dictionary with scan results
    """
    scanner = CppFileScanner(root_path)
    files = scanner.scan_directory(min_size=min_size, max_size=max_size)

    result = {
        'root_path': str(scanner.root_path),
        'total_files': len(files),
        'files': files
    }

    if organize:
        projects = scanner.organize_by_project(files)
        result['projects'] = projects
        result['project_count'] = len(projects)

    return result
