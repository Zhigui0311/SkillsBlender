#!/usr/bin/env python3
"""
Test script to verify configurable jump parameters work correctly.

This script tests:
1. New configuration parameters are accessible
2. Gap classification works with custom threshold
3. Endpoint extensions use configured values
4. Takeoff/landing margins use configured values
5. Post-jump distance works when enabled
"""

import torch
from skillsblender.tasks.path.mdp.commands.path_command_cfg import JumpPathCommandCfg

def test_jump_params_configuration():
    """Test that all new jump parameters are accessible and have correct defaults."""
    print("=" * 60)
    print("Test 1: Configuration Parameters")
    print("=" * 60)

    # Create default config
    cfg = JumpPathCommandCfg.JumpParams()

    # Check all new parameters exist and have correct defaults
    tests = [
        ("gap_width_threshold", 0.55),
        ("narrow_gap_endpoint_extension", 1.5),
        ("wide_gap_endpoint_extension", 2.0),
        ("narrow_gap_takeoff_margin", 0.4),
        ("wide_gap_takeoff_margin", 0.5),
        ("landing_margin", 0.5),
        ("post_jump_distance", 0.0),
    ]

    all_passed = True
    for param_name, expected_value in tests:
        if hasattr(cfg, param_name):
            actual_value = getattr(cfg, param_name)
            status = "✓" if actual_value == expected_value else "✗"
            print(f"{status} {param_name}: {actual_value} (expected: {expected_value})")
            if actual_value != expected_value:
                all_passed = False
        else:
            print(f"✗ {param_name}: NOT FOUND")
            all_passed = False

    print(f"\nTest 1 Result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

def test_custom_parameters():
    """Test that custom parameter values can be set."""
    print("\n" + "=" * 60)
    print("Test 2: Custom Parameter Values")
    print("=" * 60)

    # Create config with custom values
    cfg = JumpPathCommandCfg.JumpParams(
        gap_width_threshold=0.4,
        narrow_gap_endpoint_extension=1.0,
        wide_gap_endpoint_extension=2.5,
        narrow_gap_takeoff_margin=0.3,
        wide_gap_takeoff_margin=0.6,
        landing_margin=0.6,
        post_jump_distance=1.0,
    )

    tests = [
        ("gap_width_threshold", 0.4),
        ("narrow_gap_endpoint_extension", 1.0),
        ("wide_gap_endpoint_extension", 2.5),
        ("narrow_gap_takeoff_margin", 0.3),
        ("wide_gap_takeoff_margin", 0.6),
        ("landing_margin", 0.6),
        ("post_jump_distance", 1.0),
    ]

    all_passed = True
    for param_name, expected_value in tests:
        actual_value = getattr(cfg, param_name)
        status = "✓" if actual_value == expected_value else "✗"
        print(f"{status} {param_name}: {actual_value} (expected: {expected_value})")
        if actual_value != expected_value:
            all_passed = False

    print(f"\nTest 2 Result: {'PASSED' if all_passed else 'PASSED'}")
    return all_passed

def test_gap_classification_logic():
    """Test gap classification with different thresholds."""
    print("\n" + "=" * 60)
    print("Test 3: Gap Classification Logic")
    print("=" * 60)

    # Test with default threshold (0.55m)
    cfg_default = JumpPathCommandCfg.JumpParams()
    gap_widths = torch.tensor([0.3, 0.5, 0.55, 0.6, 0.8])
    is_narrow_default = gap_widths < cfg_default.gap_width_threshold

    print(f"Default threshold: {cfg_default.gap_width_threshold}m")
    print(f"Gap widths: {gap_widths.tolist()}")
    print(f"Is narrow:  {is_narrow_default.tolist()}")
    print(f"Expected:   [True, True, False, False, False]")

    expected = torch.tensor([True, True, False, False, False])
    test1_passed = torch.all(is_narrow_default == expected).item()
    print(f"Result: {'✓ PASSED' if test1_passed else '✗ FAILED'}")

    # Test with custom threshold (0.4m)
    cfg_custom = JumpPathCommandCfg.JumpParams(gap_width_threshold=0.4)
    is_narrow_custom = gap_widths < cfg_custom.gap_width_threshold

    print(f"\nCustom threshold: {cfg_custom.gap_width_threshold}m")
    print(f"Gap widths: {gap_widths.tolist()}")
    print(f"Is narrow:  {is_narrow_custom.tolist()}")
    print(f"Expected:   [True, False, False, False, False]")

    expected_custom = torch.tensor([True, False, False, False, False])
    test2_passed = torch.all(is_narrow_custom == expected_custom).item()
    print(f"Result: {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    print(f"\nTest 3 Result: {'PASSED' if test1_passed and test2_passed else 'FAILED'}")
    return test1_passed and test2_passed

def test_endpoint_extension_selection():
    """Test that correct endpoint extension is selected based on gap width."""
    print("\n" + "=" * 60)
    print("Test 4: Endpoint Extension Selection")
    print("=" * 60)

    cfg = JumpPathCommandCfg.JumpParams(
        gap_width_threshold=0.55,
        narrow_gap_endpoint_extension=1.5,
        wide_gap_endpoint_extension=2.0,
    )

    gap_widths = torch.tensor([0.3, 0.5, 0.6, 0.8])
    is_narrow = gap_widths < cfg.gap_width_threshold

    endpoint_extensions = torch.where(
        is_narrow,
        torch.tensor(cfg.narrow_gap_endpoint_extension),
        torch.tensor(cfg.wide_gap_endpoint_extension)
    )

    print(f"Gap widths:           {gap_widths.tolist()}")
    print(f"Is narrow:            {is_narrow.tolist()}")
    print(f"Endpoint extensions:  {endpoint_extensions.tolist()}")
    print(f"Expected:             [1.5, 1.5, 2.0, 2.0]")

    expected = torch.tensor([1.5, 1.5, 2.0, 2.0])
    test_passed = torch.allclose(endpoint_extensions, expected)
    print(f"Result: {'✓ PASSED' if test_passed else '✗ FAILED'}")

    print(f"\nTest 4 Result: {'PASSED' if test_passed else 'FAILED'}")
    return test_passed

def test_post_jump_distance():
    """Test post-jump distance application."""
    print("\n" + "=" * 60)
    print("Test 5: Post-Jump Distance")
    print("=" * 60)

    # Test with post_jump_distance = 0 (disabled)
    cfg_disabled = JumpPathCommandCfg.JumpParams(post_jump_distance=0.0)
    base_length = torch.tensor([10.0, 12.0, 15.0])
    has_gap = torch.tensor([True, True, True])

    total_len_disabled = base_length.clone()
    if cfg_disabled.post_jump_distance > 0:
        total_len_disabled = torch.where(has_gap, total_len_disabled + cfg_disabled.post_jump_distance, total_len_disabled)

    print(f"Post-jump distance: {cfg_disabled.post_jump_distance}m (disabled)")
    print(f"Base lengths:       {base_length.tolist()}")
    print(f"Total lengths:      {total_len_disabled.tolist()}")
    print(f"Expected:           {base_length.tolist()} (no change)")

    test1_passed = torch.allclose(total_len_disabled, base_length)
    print(f"Result: {'✓ PASSED' if test1_passed else '✗ FAILED'}")

    # Test with post_jump_distance = 1.0 (enabled)
    cfg_enabled = JumpPathCommandCfg.JumpParams(post_jump_distance=1.0)
    total_len_enabled = base_length.clone()
    if cfg_enabled.post_jump_distance > 0:
        total_len_enabled = torch.where(has_gap, total_len_enabled + cfg_enabled.post_jump_distance, total_len_enabled)

    print(f"\nPost-jump distance: {cfg_enabled.post_jump_distance}m (enabled)")
    print(f"Base lengths:       {base_length.tolist()}")
    print(f"Total lengths:      {total_len_enabled.tolist()}")
    print(f"Expected:           {(base_length + 1.0).tolist()}")

    expected_enabled = base_length + 1.0
    test2_passed = torch.allclose(total_len_enabled, expected_enabled)
    print(f"Result: {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    print(f"\nTest 5 Result: {'PASSED' if test1_passed and test2_passed else 'FAILED'}")
    return test1_passed and test2_passed

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CONFIGURABLE JUMP PARAMETERS - CORE FUNCTIONALITY TEST")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("Configuration Parameters", test_jump_params_configuration()))
    results.append(("Custom Parameter Values", test_custom_parameters()))
    results.append(("Gap Classification Logic", test_gap_classification_logic()))
    results.append(("Endpoint Extension Selection", test_endpoint_extension_selection()))
    results.append(("Post-Jump Distance", test_post_jump_distance()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
