#!/usr/bin/env python3
"""
Verification script to ensure repository is ready for GitHub
"""

import pathlib
import json
import sys

def check_file_exists(path, description):
    """Check if a file exists and report status"""
    if pathlib.Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} - MISSING")
        return False

def check_directory_exists(path, description):
    """Check if a directory exists and report status"""
    if pathlib.Path(path).is_dir():
        print(f"✅ {description}: {path}/")
        return True
    else:
        print(f"❌ {description}: {path}/ - MISSING")
        return False

def check_file_size(path, max_size_kb, description):
    """Check if file is under size limit"""
    file_path = pathlib.Path(path)
    if file_path.exists():
        size_kb = file_path.stat().st_size / 1024
        if size_kb <= max_size_kb:
            print(f"✅ {description}: {path} ({size_kb:.1f} KB)")
            return True
        else:
            print(f"⚠️  {description}: {path} ({size_kb:.1f} KB > {max_size_kb} KB)")
            return False
    else:
        print(f"❌ {description}: {path} - MISSING")
        return False

def main():
    print("🔍 GitHub Repository Readiness Check")
    print("=" * 40)
    
    all_good = True
    
    # Core files
    print("\n📁 Core Files:")
    all_good &= check_file_exists("README.md", "Main README")
    all_good &= check_file_exists("LICENSE", "License file")
    all_good &= check_file_exists("CMakeLists.txt", "CMake build file")
    all_good &= check_file_exists(".gitignore", "Git ignore file")
    all_good &= check_file_exists("CONTRIBUTING.md", "Contributing guide")
    
    # Source code
    print("\n💻 Source Code:")
    all_good &= check_directory_exists("cpu", "CPU implementations")
    all_good &= check_directory_exists("include", "Header files")
    all_good &= check_directory_exists("bench", "Benchmark harness")
    all_good &= check_directory_exists("tests", "Test suite")
    all_good &= check_directory_exists("baselines", "Baseline implementations")
    
    # Scripts
    print("\n🔧 Scripts:")
    all_good &= check_file_exists("scripts/bench.ps1", "Windows benchmark script")
    all_good &= check_file_exists("scripts/bench.sh", "Linux/macOS benchmark script")
    all_good &= check_file_exists("scripts/plot.py", "Plotting script")
    all_good &= check_file_exists("scripts/verify_plots.py", "Plot verification")
    
    # Documentation
    print("\n📚 Documentation:")
    all_good &= check_directory_exists("docs", "Documentation directory")
    all_good &= check_file_exists("docs/BUILD_INSTRUCTIONS.md", "Build instructions")
    all_good &= check_file_exists("docs/ARCHITECTURE.md", "Architecture docs")
    all_good &= check_file_exists("docs/PERFORMANCE_RESULTS.md", "Performance analysis")
    
    # Data files (small)
    print("\n📊 Data Files:")
    all_good &= check_file_size("data/best_tiles.json", 5, "Tile configurations")
    all_good &= check_file_size("data/example_results.csv", 2, "Example results")
    
    # Hero plot
    print("\n🖼️  Showcase Assets:")
    all_good &= check_file_exists("results/plots/gemm_gflops_vs_N.png", "Hero plot")
    
    # CI/CD
    print("\n🔄 CI/CD:")
    all_good &= check_file_exists(".github/workflows/ci.yml", "GitHub Actions")
    
    # Development tools
    print("\n🛠️  Development Tools:")
    all_good &= check_file_exists(".clang-format", "Code formatting")
    all_good &= check_file_exists(".editorconfig", "Editor config")
    all_good &= check_file_exists(".vscode/settings.json", "VS Code settings")
    
    # Check for files that shouldn't be there
    print("\n🚫 Unwanted Files Check:")
    unwanted_patterns = [
        "*.obj", "*.exe", "*.pdb", "*.ilk",  # Build artifacts
        "build/", "out/", "Debug/", "Release/",  # Build directories
        "data/runs/",  # Large result files
        "__pycache__/", "*.pyc",  # Python cache
    ]
    
    unwanted_found = []
    for pattern in unwanted_patterns:
        matches = list(pathlib.Path(".").glob(pattern))
        if matches:
            unwanted_found.extend(matches)
    
    if unwanted_found:
        print("⚠️  Found unwanted files (should be in .gitignore):")
        for file in unwanted_found[:5]:  # Show first 5
            print(f"   - {file}")
        if len(unwanted_found) > 5:
            print(f"   ... and {len(unwanted_found) - 5} more")
    else:
        print("✅ No unwanted files found")
    
    # Final status
    print("\n" + "=" * 40)
    if all_good and not unwanted_found:
        print("🎉 REPOSITORY IS READY FOR GITHUB!")
        print("✅ All required files present")
        print("✅ No unwanted files detected")
        print("✅ File sizes appropriate")
        print("\n🚀 Ready to push with:")
        print("   git add .")
        print("   git commit -m 'High-performance GEMM: 360.39 GFLOP/s'")
        print("   git push origin main")
        return True
    else:
        print("❌ REPOSITORY NOT READY")
        print("Please fix the issues above before pushing to GitHub")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)