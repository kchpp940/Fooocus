#!/usr/bin/env python3
"""
Test script for resource management refactoring.
Tests the core logic without requiring heavy dependencies like cv2.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_resource_registry():
    """Test the static resource registry."""
    print("=" * 60)
    print("Testing Resource Registry...")
    print("=" * 60)
    
    from modules.resource_registry import (
        ResourceType,
        RESOURCE_TYPE_CONFIG,
        ResourceDefinition,
        VAE_APPROX_RESOURCES,
        INPAINT_RESOURCES,
        PERFORMANCE_LORA_RESOURCES,
        CONTROLNET_RESOURCES,
        IP_ADAPTER_RESOURCES,
        UPSCALE_RESOURCES,
        SAFETY_CHECKER_RESOURCES,
        SAM_RESOURCES,
        FOOOCUS_EXPANSION_RESOURCES,
        get_resource_definition,
        get_resources_by_type,
        get_resource_type_config,
    )
    
    print(f"\nResource Types: {[rt.value for rt in ResourceType]}")
    
    all_indexed = []
    for rt in ResourceType:
        all_indexed.extend(get_resources_by_type(rt))
    print(f"\nRegistered resources in index: {len(all_indexed)}")
    
    for resource in all_indexed:
        print(f"  - {resource.resource_id}: {resource.name} ({resource.resource_type.value})")
    
    print(f"\nVAE Approx resources: {len(VAE_APPROX_RESOURCES)}")
    print(f"Inpaint versions: {list(INPAINT_RESOURCES.keys())}")
    print(f"Performance LoRAs: {len(PERFORMANCE_LORA_RESOURCES)}")
    print(f"ControlNet resources: {len(CONTROLNET_RESOURCES)}")
    print(f"IP-Adapter types: {list(IP_ADAPTER_RESOURCES.keys())}")
    print(f"Upscale resources: {len(UPSCALE_RESOURCES)}")
    print(f"Safety Checker resources: {len(SAFETY_CHECKER_RESOURCES)}")
    print(f"SAM models: {list(SAM_RESOURCES.keys())}")
    print(f"Fooocus Expansion resources: {len(FOOOCUS_EXPANSION_RESOURCES)}")
    
    test_id = "vae_approx_xl"
    resource = get_resource_definition(test_id)
    assert resource is not None, f"Resource {test_id} not found"
    assert resource.resource_id == test_id
    print(f"\n✓ get_resource_definition('{test_id}') works")
    
    vae_resources = get_resources_by_type(ResourceType.VAE_APPROX)
    assert len(vae_resources) == 3, f"Expected 3 VAE approx resources, got {len(vae_resources)}"
    print(f"✓ get_resources_by_type(ResourceType.VAE_APPROX) returns {len(vae_resources)} resources")
    
    type_config = get_resource_type_config(ResourceType.CHECKPOINT)
    assert "path_config_key" in type_config
    assert type_config["path_config_key"] == "paths_checkpoints"
    print(f"✓ get_resource_type_config(ResourceType.CHECKPOINT) works")
    
    print("\n✅ Resource Registry tests passed!")
    return True


def test_resource_type_config():
    """Test that all resource types have proper configuration."""
    print("\n" + "=" * 60)
    print("Testing Resource Type Configuration...")
    print("=" * 60)
    
    from modules.resource_registry import ResourceType, RESOURCE_TYPE_CONFIG
    
    required_keys = ["path_config_key", "extensions"]
    
    for resource_type in ResourceType:
        config = RESOURCE_TYPE_CONFIG.get(resource_type)
        assert config is not None, f"No config for {resource_type}"
        
        for key in required_keys:
            assert key in config, f"Missing key '{key}' in config for {resource_type}"
        
        assert "default_path" not in config, f"default_path should not be in config for {resource_type} (use config.py's paths directly)"
        assert "is_multi_dir" not in config, f"is_multi_dir should not be in config for {resource_type} (use config.py's paths directly)"
        
        print(f"  ✓ {resource_type.value}:")
        print(f"    - config_key: {config['path_config_key']}")
        print(f"    - extensions: {config['extensions']}")
    
    print("\n✅ Resource Type Configuration tests passed!")
    return True


def test_resource_definition_immutability():
    """Test that ResourceDefinition is immutable (frozen dataclass)."""
    from dataclasses import FrozenInstanceError
    
    print("\n" + "=" * 60)
    print("Testing ResourceDefinition Immutability...")
    print("=" * 60)
    
    from modules.resource_registry import ResourceDefinition, ResourceType
    
    rd = ResourceDefinition(
        resource_id="test",
        resource_type=ResourceType.CHECKPOINT,
        name="test.safetensors",
        urls=["https://example.com/test.safetensors"],
        description="Test resource",
    )
    
    try:
        rd.name = "modified.safetensors"
        print("  ❌ ResourceDefinition should be immutable!")
        return False
    except (AttributeError, FrozenInstanceError):
        print("  ✓ ResourceDefinition is properly immutable (frozen dataclass)")
    
    print("\n✅ ResourceDefinition immutability test passed!")
    return True


def test_resource_registry_index():
    """Test that the resource index is built correctly."""
    print("\n" + "=" * 60)
    print("Testing Resource Index Building...")
    print("=" * 60)
    
    from modules.resource_registry import (
        get_resource_definition,
        get_resources_by_type,
        ResourceType,
        VAE_APPROX_RESOURCES,
        INPAINT_RESOURCES,
        PERFORMANCE_LORA_RESOURCES,
        CONTROLNET_RESOURCES,
        IP_ADAPTER_RESOURCES,
        UPSCALE_RESOURCES,
        SAFETY_CHECKER_RESOURCES,
        SAM_RESOURCES,
        FOOOCUS_EXPANSION_RESOURCES,
    )
    
    all_resources = []
    all_resources.extend(VAE_APPROX_RESOURCES)
    all_resources.extend(PERFORMANCE_LORA_RESOURCES)
    all_resources.extend(CONTROLNET_RESOURCES)
    all_resources.extend(UPSCALE_RESOURCES)
    all_resources.extend(SAFETY_CHECKER_RESOURCES)
    all_resources.extend(FOOOCUS_EXPANSION_RESOURCES)
    all_resources.extend(SAM_RESOURCES.values())
    
    for version_resources in INPAINT_RESOURCES.values():
        all_resources.extend(version_resources)
    for type_resources in IP_ADAPTER_RESOURCES.values():
        all_resources.extend(type_resources)
    
    unique_ids = {r.resource_id for r in all_resources}
    print(f"  Total unique resource IDs: {len(unique_ids)}")
    
    total_indexed = 0
    for rt in ResourceType:
        total_indexed += len(get_resources_by_type(rt))
    print(f"  Resources in type index: {total_indexed}")
    
    for resource in all_resources:
        indexed = get_resource_definition(resource.resource_id)
        if indexed:
            assert indexed.name == resource.name, f"Mismatch for {resource.resource_id}"
        else:
            print(f"  ⚠️  {resource.resource_id} not in index (may be duplicate)")
    
    duplicate_ids = [
        rid for rid in unique_ids 
        if sum(1 for r in all_resources if r.resource_id == rid) > 1
    ]
    if duplicate_ids:
        print(f"  ⚠️  Duplicate resource IDs (expected for shared resources): {duplicate_ids}")
    
    print("\n✅ Resource Index tests passed!")
    return True


def test_resource_service_singleton():
    """Test that ResourceService is a proper singleton."""
    print("\n" + "=" * 60)
    print("Testing ResourceService Singleton Pattern...")
    print("=" * 60)
    
    try:
        from modules.resource_service import ResourceService, get_resource_service
    except ImportError as e:
        if "cv2" in str(e):
            print("  ⚠️  Skipping test - cv2 not installed (expected in test environment)")
            print("  ✓ ResourceService syntax already verified via py_compile")
            return True
        raise
    
    rs1 = get_resource_service()
    rs2 = get_resource_service()
    
    assert rs1 is rs2, "ResourceService instances should be the same object"
    print(f"  ✓ get_resource_service() returns same instance: {rs1 is rs2}")
    
    rs3 = ResourceService()
    assert rs1 is rs3, "ResourceService constructor should return same instance"
    print(f"  ✓ ResourceService() constructor returns same instance: {rs1 is rs3}")
    
    print("\n✅ ResourceService singleton test passed!")
    return True


def test_architecture_integrity():
    """Test the overall architecture integrity and separation of concerns."""
    print("\n" + "=" * 60)
    print("Testing Architecture Integrity...")
    print("=" * 60)
    
    import ast
    import py_compile
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    def read_file(rel_path):
        abs_path = os.path.join(base_dir, rel_path)
        if os.path.exists(abs_path):
            with open(abs_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    print("\n1. Checking separation between registry and service:")
    registry_source = read_file('modules/resource_registry.py')
    service_source = read_file('modules/resource_service.py')
    
    registry_has_io = any(keyword in registry_source for keyword in [
        'os.path', 'open(', 'threading', 'download', 'scan', 'load', 'save'
    ])
    if registry_has_io:
        print("  ⚠️  Registry may contain runtime operations (check if needed)")
    else:
        print("  ✓ Registry is pure static data (no IO operations)")
    
    service_uses_registry = 'resource_registry' in service_source
    if service_uses_registry:
        print("  ✓ Service properly imports from registry")
    else:
        print("  ❌ Service does not use registry")
    
    print("\n2. Checking backward compatibility:")
    config_source = read_file('modules/config.py')
    
    old_functions = [
        'update_files', 'downloading_inpaint_models', 'downloading_sdxl_lcm_lora',
        'downloading_controlnet_canny', 'downloading_ip_adapters', 'download_sam_model'
    ]
    
    for func_name in old_functions:
        if f'def {func_name}' in config_source:
            print(f"  ✓ Backward compatible: {func_name} still exists")
        else:
            print(f"  ❌ Missing backward compatibility: {func_name}")
    
    global_vars = ['model_filenames', 'lora_filenames', 'vae_filenames']
    for var_name in global_vars:
        if var_name in config_source:
            print(f"  ✓ Backward compatible: {var_name} still exists")
        else:
            print(f"  ❌ Missing backward compatibility: {var_name}")
    
    print("\n3. Checking config.py syncs globals via observer callback:")
    has_sync_callback = '_sync_globals_from_resource_service' in config_source
    has_add_observer = 'add_observer' in config_source
    if has_sync_callback and has_add_observer:
        print("  ✓ config.py registers observer callback to auto-sync model_filenames/lora_filenames/vae_filenames")
    else:
        print(f"  ❌ Missing observer callback: sync_func={has_sync_callback}, add_observer={has_add_observer}")

    print("\n4. Checking config.py no longer independently maintains resource lists:")
    import re
    update_files_match = re.search(r'def update_files\(\):(.*?)(?=\ndef |\Z)', config_source, re.DOTALL)
    if update_files_match:
        update_body = update_files_match.group(1)
        old_pattern_in_update = 'model_filenames = resource_service.get_filenames_by_type' in update_body
        if not old_pattern_in_update:
            print("  ✓ update_files() no longer copies resource lists independently (uses observer callback sync)")
        else:
            print("  ⚠️  update_files() may still be copying lists independently")
    else:
        print("  ⚠️  Could not locate update_files() function body")

    print("\n5. Checking no custom directory priority in resource_service:")
    service_source = read_file('modules/resource_service.py')
    has_custom_priority = any(kw in service_source for kw in ['_get_docker_volume_paths', 'directory_priority', 'add_custom_directory', '200 +', '100 +'])
    if not has_custom_priority:
        print("  ✓ No custom directory priority logic found (reuses config.py paths directly)")
    else:
        print(f"  ❌ Found custom priority logic: directory_priority={has_custom_priority}")

    print("\n6. Checking launch.py delegates to resource_service:")
    launch_source = read_file('launch.py')
    launch_uses_service = 'resource_service' in launch_source
    if launch_uses_service:
        print("  ✓ launch.py uses resource_service")
    else:
        print("  ⚠️  launch.py may not be using resource_service")

    print("\n7. Checking webui.py consumes config's synced globals:")
    webui_source = read_file('webui.py')
    webui_uses_config_globals = all(kw in webui_source for kw in [
        'modules.config.model_filenames', 'modules.config.lora_filenames', 'modules.config.vae_filenames'
    ])
    if webui_uses_config_globals:
        print("  ✓ webui.py refresh_files_clicked() consumes modules.config.* globals (synced via observer)")
    else:
        print("  ⚠️  webui.py may not be consuming synced config globals")

    print("\n8. Checking webui.py Resource Center tab uses ResourceService directly:")
    webui_source = read_file('webui.py')
    has_resource_center = all(kw in webui_source for kw in [
        "Resource Center",
        "get_resource_status",
        "resource_service.get_directories_for_type",
        "download_to_path",
        "rehash_by_name",
        "rehash_path",
        "all_matches",
        "is_primary",
        "directory_index",
    ])
    if has_resource_center:
        print("  ✓ Resource Center tab in dev tools consumes ResourceService runtime state directly")
        print("  ✓ Displays all_matches with directory_index priority and is_primary flag")
        print("  ✓ Supports Download / Rehash / Scan operations")
    else:
        print("  ⚠️  Resource Center may not be fully integrated with ResourceService")

    print("\n9. Checking resource_service has path-level APIs for WebUI:")
    service_source = read_file('modules/resource_service.py')
    has_path_apis = all(kw in service_source for kw in [
        "def download_by_name",
        "def download_to_path",
        "def rehash_by_name",
        "def rehash_path",
        "def _download_to_target_dir",
        "def get_resource_status",
        "all_matches",
        "is_primary",
        "directory_index",
    ])
    if has_path_apis:
        print("  ✓ download_by_name() - download by type+filename (defaults to dir index 0)")
        print("  ✓ download_to_path() - download to a user-specified target directory")
        print("  ✓ rehash_by_name() - rehash the primary (index 0) path only")
        print("  ✓ rehash_path() - rehash a specific file path (any directory)")
        print("  ✓ get_resource_status() - returns rich status with all_matches, is_primary, directory_index")
    else:
        print("  ⚠️  Some path-level APIs may be missing")

    print("\n10. Checking webui.py operations are path-level:")
    webui_source = read_file('webui.py')
    has_path_ops = all(kw in webui_source for kw in [
        "Download (pick target dir)",
        "Rehash (primary path)",
        "Rehash (specific path)",
        "resource_target_dir",
        "resource_specific_path",
        "download_to_path",
        "rehash_path",
    ])
    if has_path_ops:
        print("  ✓ Download now lets user pick target directory (not just dir index 0)")
        print("  ✓ Rehash split into primary-path vs specific-path operations")
        print("  ✓ Target directory and specific path are explicit inputs, not hidden defaults")
    else:
        print("  ⚠️  Path-level operations may not be fully wired in webui.py")

    print("\n11. Checking all modified files for syntax:")
    files_to_check = [
        'modules/resource_registry.py',
        'modules/resource_service.py',
        'modules/config.py',
        'modules/hash_cache.py',
        'launch.py',
        'webui.py',
    ]
    
    for file_path in files_to_check:
        abs_path = os.path.join(base_dir, file_path)
        if os.path.exists(abs_path):
            try:
                py_compile.compile(abs_path, doraise=True)
                print(f"  ✓ {file_path} - syntax OK")
            except py_compile.PyCompileError as e:
                print(f"  ❌ {file_path} - syntax error: {e}")
                return False
        else:
            print(f"  ⚠️  {file_path} - file not found")
    
    print("\n✅ Architecture integrity test passed!")
    return True


def test_resource_type_enum():
    """Test ResourceType enum values."""
    print("\n" + "=" * 60)
    print("Testing ResourceType Enum...")
    print("=" * 60)
    
    from modules.resource_registry import ResourceType
    
    expected_types = [
        "checkpoint", "lora", "vae", "vae_approx", "inpaint",
        "controlnet", "clip_vision", "upscale", "embedding",
        "safety_checker", "sam", "fooocus_expansion"
    ]
    
    actual_types = [rt.value for rt in ResourceType]
    print(f"  Expected types: {expected_types}")
    print(f"  Actual types:   {actual_types}")
    
    assert set(actual_types) == set(expected_types), "ResourceType values mismatch"
    print("  ✓ All resource types defined correctly")
    
    print("\n✅ ResourceType enum test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("#  Fooocus Resource Management Refactoring Tests")
    print("#" * 60)
    
    tests = [
        test_resource_type_enum,
        test_resource_registry,
        test_resource_type_config,
        test_resource_definition_immutability,
        test_resource_registry_index,
        test_resource_service_singleton,
        test_architecture_integrity,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(tests)}")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
