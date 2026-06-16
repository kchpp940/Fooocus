#!/usr/bin/env python3
"""
Verify unified user data directory structure.
User presets and sorted styles should share the same root directory.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_unified_user_data_dir():
    import modules.config as config

    user_data_dir = config.get_user_data_dir()
    user_presets_dir = config.get_user_presets_dir()
    sorted_styles_path = config.get_sorted_styles_path()

    print(f"path_user_data (config_dict): {config.config_dict.get('path_user_data')}")
    print(f"get_user_data_dir():          {user_data_dir}")
    print(f"get_user_presets_dir():       {user_presets_dir}")
    print(f"get_sorted_styles_path():     {sorted_styles_path}")

    assert os.path.isabs(user_data_dir), "User data dir should be absolute"
    assert os.path.exists(user_data_dir), "User data dir should exist"
    assert os.path.exists(user_presets_dir), "User presets dir should exist"

    presets_parent = os.path.dirname(user_presets_dir)
    styles_parent = os.path.dirname(sorted_styles_path)

    assert presets_parent == user_data_dir, \
        f"User presets should be under user_data_dir: {presets_parent} != {user_data_dir}"
    assert styles_parent == user_data_dir, \
        f"Sorted styles should be under user_data_dir: {styles_parent} != {user_data_dir}"

    print(f"\n[PASS] Both user_presets/ and sorted_styles.json are under: {user_data_dir}")


def test_style_sorter_uses_config_path():
    from modules import style_sorter

    sorted_styles_path = style_sorter._sorted_styles_path
    import modules.config as config
    expected_path = config.get_sorted_styles_path()

    print(f"\nstyle_sorter._sorted_styles_path: {sorted_styles_path}")
    print(f"config.get_sorted_styles_path(): {expected_path}")

    assert sorted_styles_path == expected_path, \
        f"style_sorter path should match config: {sorted_styles_path} != {expected_path}"

    print("[PASS] style_sorter uses unified path from config")


def test_config_dict_has_path_user_data():
    import modules.config as config

    assert 'path_user_data' in config.config_dict, \
        "config_dict should have path_user_data key"

    path_from_dict = config.config_dict['path_user_data']
    path_from_func = config.get_user_data_dir()

    print(f"\nconfig_dict['path_user_data']: {path_from_dict}")
    print(f"get_user_data_dir():           {path_from_func}")

    assert path_from_dict == path_from_func, \
        f"config_dict value should match function return: {path_from_dict} != {path_from_func}"

    print("[PASS] config_dict has path_user_data synced with get_user_data_dir()")


def test_env_var_override():
    import modules.config as config

    orig_cache = config._user_data_dir
    config._user_data_dir = None
    os.environ['FOOOCUS_USER_DATA_DIR'] = '/tmp/test_fooocus_user_data'

    try:
        user_dir = config.get_user_data_dir()
        assert user_dir == '/tmp/test_fooocus_user_data', \
            f"FOOOCUS_USER_DATA_DIR env var should override: {user_dir}"
        print(f"\n[PASS] FOOOCUS_USER_DATA_DIR env var override works: {user_dir}")
    finally:
        del os.environ['FOOOCUS_USER_DATA_DIR']
        config._user_data_dir = orig_cache


def test_path_user_data_env_var():
    import modules.config as config

    orig_cache = config._user_data_dir
    config._user_data_dir = None
    os.environ['path_user_data'] = '/tmp/test_path_user_data'

    try:
        user_dir = config.get_user_data_dir()
        assert user_dir == '/tmp/test_path_user_data', \
            f"path_user_data env var should work: {user_dir}"
        print(f"[PASS] path_user_data env var works: {user_dir}")
    finally:
        del os.environ['path_user_data']
        config._user_data_dir = orig_cache


def test_sorted_styles_env_backward_compat():
    import modules.config as config

    os.environ['sorted_styles_path'] = '/tmp/custom_sorted_styles.json'

    try:
        path = config.get_sorted_styles_path()
        assert path == '/tmp/custom_sorted_styles.json', \
            f"sorted_styles_path env var should override: {path}"
        print(f"[PASS] sorted_styles_path env backward compat works: {path}")
    finally:
        del os.environ['sorted_styles_path']


def main():
    print("=" * 70)
    print("Unified User Data Directory - Consistency Tests")
    print("=" * 70)
    print()

    tests = [
        test_unified_user_data_dir,
        test_config_dict_has_path_user_data,
        test_style_sorter_uses_config_path,
        test_env_var_override,
        test_path_user_data_env_var,
        test_sorted_styles_env_backward_compat,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"\n[ERROR] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
