import os
import subprocess
import sys

# Configuration
LOCAL_FOLDER = "/content/tensormorph_local"
OPT_TOOL = f"{LOCAL_FOLDER}/build/tensormorph-opt"
FILECHECK = "/usr/lib/llvm-18/bin/FileCheck"
TEST_DIR = f"{LOCAL_FOLDER}/tests"

def run_process(cmd):
    """Utility to run a shell command and return output."""
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode

def run_flag_tests():
    """Suite of tests for CLI flags and high-level logic."""
    print("\n[1/2] Running Flag Logic Tests...")
    passed = 0
    total = 6
    
    # Case 1: Transpose Folding.
    out, _, _ = run_process(f'{OPT_TOOL} --tosa-opt {TEST_DIR}/transpose_fold.mlir')
    out_off, _, _ = run_process(f'{OPT_TOOL} --tosa-opt="fuse-transpose=false" {TEST_DIR}/transpose_fold.mlir')
    if "tosa.transpose" not in out and "tosa.transpose" in out_off:
        print("  [OK] Transpose toggle")
        passed += 1
    else:
        print("  [FAIL] Transpose toggle")

    # Case 2: Padding Elimination.
    out, _, _ = run_process(f'{OPT_TOOL} --tosa-opt {TEST_DIR}/pad_elimination.mlir')
    out_off, _, _ = run_process(f'{OPT_TOOL} --tosa-opt="fuse-padding=false" {TEST_DIR}/pad_elimination.mlir')
    if "tosa.pad" not in out and "tosa.pad" in out_off:
        print("  [OK] Padding toggle")
        passed += 1
    else:
        print("  [FAIL] Padding toggle")

    # Case 3: Linear Math Folding.
    out, _, _ = run_process(f'{OPT_TOOL} --tosa-opt {TEST_DIR}/conv_add_clamp.mlir')
    out_off, _, _ = run_process(f'{OPT_TOOL} --tosa-opt="fuse-linear=false" {TEST_DIR}/conv_add_clamp.mlir')
    if "tosa.add" not in out and "tosa.add" in out_off:
        print("  [OK] Linear Math toggle")
        passed += 1
    else:
        print("  [FAIL] Linear Math toggle")

    # Case 4: Fan-out Cloning.
    out, _, _ = run_process(f'{OPT_TOOL} --tosa-opt {TEST_DIR}/fanout_cloning.mlir')
    if out.count("tosa.conv2d") == 2:
        print("  [OK] Fan-out cloning")
        passed += 1
    else:
        print("  [FAIL] Fan-out cloning")

    # Case 5: AI Mock Approval logic.
    # Score is 1.0. min-profit 0.5 < 1.0 (Fusion should happen).
    out, _, _ = run_process(f'{OPT_TOOL} --tosa-opt="ai-advisor=mock min-profit=0.5" {TEST_DIR}/conv_add_clamp.mlir')
    if "tosa.add" not in out:
        print("  [OK] AI Mock Approval logic")
        passed += 1
    else:
        print("  [FAIL] AI Mock Approval logic")

    # Case 6: AI Mock Veto logic.
    # Score is 1.0. min-profit 1.5 > 1.0 (Veto should happen).
    out, _, _ = run_process(f'{OPT_TOOL} --tosa-opt="ai-advisor=mock min-profit=1.5" {TEST_DIR}/conv_add_clamp.mlir')
    if "tosa.add" in out:
        print("  [OK] AI Mock Veto logic")
        passed += 1
    else:
        print("  [FAIL] AI Mock Veto logic")

    return passed, total

def run_regression_tests():
    """Suite of MLIR tests validated via FileCheck."""
    print("\n[2/2] Running MLIR Regression Suite...")
    if not os.path.exists(TEST_DIR):
        print(f"  [ERROR] Test directory not found: {TEST_DIR}")
        return 0, 0

    tests = [f for f in os.listdir(TEST_DIR) if f.endswith(".mlir")]
    passed = 0

    for test_file in sorted(tests):
        test_path = os.path.join(TEST_DIR, test_file)
        cmd = f"{OPT_TOOL} --tosa-opt {test_path} | {FILECHECK} {test_path}"
        _, stderr, ret = run_process(cmd)

        if ret == 0:
            print(f"  [OK] {test_file}")
            passed += 1
        else:
            print(f"  [FAIL] {test_file}")
            if stderr: print(f"       Error: {stderr.strip()}")

    return passed, len(tests)

if __name__ == "__main__":
    print("=" * 40)
    print(" TENSORMORPH TEST RUNNER")
    print("=" * 40)

    f_pass, f_total = run_flag_tests()
    r_pass, r_total = run_regression_tests()
    
    print("\n" + "=" * 40)
    if f_pass == f_total and r_pass == r_total:
        print(f" SUCCESS: All {f_total + r_total} tests passed!")
        print("=" * 40)
        sys.exit(0)
    else:
        print(f" FAILURE: {f_total - f_pass} flag tests and {r_total - r_pass} regressions failed.")
        print("=" * 40)
        sys.exit(1)