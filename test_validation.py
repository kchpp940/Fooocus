#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/pkcha/Fooocus')
from modules.util import (
    parse_prompt_matrix_config, validate_matrix_config
)

print('=' * 60)
print('Validation Test Suite')
print('=' * 60)

passed = 0
failed = 0

def test(name, config_text, max_combos=512, max_vars=20, image_number=1,
         expect_valid=True, expect_errors=0, expect_warnings=0,
         error_contains=None, warning_contains=None):
    global passed, failed
    config = parse_prompt_matrix_config(config_text)
    result = validate_matrix_config(config, max_combos, max_vars, image_number)

    ok = True
    if result['valid'] != expect_valid:
        print(f'  FAIL: expected valid={expect_valid}, got {result["valid"]}')
        ok = False
    if len(result.get('errors', [])) != expect_errors:
        print(f'  FAIL: expected {expect_errors} errors, got {len(result.get("errors", []))}: {result.get("errors", [])}')
        ok = False
    if len(result.get('warnings', [])) != expect_warnings:
        print(f'  FAIL: expected {expect_warnings} warnings, got {len(result.get("warnings", []))}: {result.get("warnings", [])}')
        ok = False
    if error_contains:
        errors_text = ' '.join(result.get('errors', []))
        if error_contains not in errors_text:
            print(f'  FAIL: errors do not contain "{error_contains}": {result.get("errors", [])}')
            ok = False
    if warning_contains:
        warnings_text = ' '.join(result.get('warnings', []))
        if warning_contains not in warnings_text:
            print(f'  FAIL: warnings do not contain "{warning_contains}": {result.get("warnings", [])}')
            ok = False

    if ok:
        passed += 1
        print(f'  PASS: {name}')
    else:
        failed += 1
        print(f'  FAIL: {name}')
    return ok

# --- Basic validation tests ---
print('\n--- Basic Validation ---')

test('Empty config is valid', '', expect_valid=True, expect_errors=0)

test('Simple text vars are valid',
     'subject: cat, dog\nquality: good, bad',
     expect_valid=True, expect_errors=0, expect_warnings=0)

test('Max combinations exceeded',
     'v1: ' + ', '.join([f'x{i}' for i in range(30)]) + '\nv2: ' + ', '.join([f'y{i}' for i in range(30)]),
     max_combos=100,
     expect_valid=False, expect_errors=1, error_contains='Too many matrix combinations')

test('Max variables exceeded',
     '\n'.join([f'v{i}: a, b' for i in range(25)]),
     max_vars=10,
     expect_valid=False, expect_errors=1, error_contains='Too many variables')

test('Empty variable name',
     ': a, b',
     expect_valid=False, expect_errors=1, error_contains='empty name')

test('Empty values list',
     'myvar: ',
     expect_valid=False, expect_errors=1, error_contains='no values')

test('Empty value in list',
     'myvar: a, , b',
     expect_valid=False, expect_errors=1, error_contains='empty values')

test('Single variable 101 values triggers warning',
     'v1: ' + ', '.join([f'x{i}' for i in range(101)]),
     expect_valid=True, expect_errors=0, expect_warnings=1, warning_contains='may take very long')

# --- Seed validation ---
print('\n--- Seed Validation ---')

test('Valid seed values',
     '@seed: 12345, 67890',
     expect_valid=True, expect_errors=0)

test('Negative seed error',
     '@seed: -1, 100',
     expect_valid=False, expect_errors=1, error_contains='out of valid range')

test('Seed too large',
     '@seed: 9999999999, 100',
     expect_valid=False, expect_errors=1, error_contains='out of valid range')

test('Non-numeric seed',
     '@seed: abc, 100',
     expect_valid=False, expect_errors=1, error_contains='invalid seed value')

# --- CFG validation ---
print('\n--- CFG Validation ---')

test('Valid cfg values',
     '@cfg: 3.5, 7.5, 10.0',
     expect_valid=True, expect_errors=0)

test('Negative cfg warning',
     '@cfg: -1.0, 5.0',
     expect_valid=True, expect_errors=0, expect_warnings=1, warning_contains='outside typical range')

test('Very high cfg warning',
     '@cfg: 100.0, 5.0',
     expect_valid=True, expect_errors=0, expect_warnings=1)

test('Non-numeric cfg error',
     '@cfg: abc, 5.0',
     expect_valid=False, expect_errors=1, error_contains='invalid cfg value')

# --- Steps validation ---
print('\n--- Steps Validation ---')

test('Valid steps',
     '@steps: 20, 30, 50',
     expect_valid=True, expect_errors=0)

test('Zero steps warning',
     '@steps: 0, 20',
     expect_valid=True, expect_errors=0, expect_warnings=1, warning_contains='outside typical range')

test('Non-integer steps error',
     '@steps: abc, 20',
     expect_valid=False, expect_errors=1, error_contains='invalid steps value')

test('Very high steps warning',
     '@steps: 500, 20',
     expect_valid=True, expect_errors=0, expect_warnings=1)

# --- Width/Height validation ---
print('\n--- Width/Height Validation ---')

test('Valid dimensions',
     '@width: 832, 1024\n@height: 1216, 1024',
     expect_valid=True, expect_errors=0)

test('Non-multiple of 8 warning',
     '@width: 1000, 1024',
     expect_valid=True, expect_errors=0, expect_warnings=1, warning_contains='not a multiple of 8')

test('Very small dimension warning',
     '@height: 64, 512',
     expect_valid=True, expect_errors=0, expect_warnings=1)

test('Very large dimension warning',
     '@width: 8192, 1024',
     expect_valid=True, expect_errors=0, expect_warnings=1)

test('Non-numeric width error',
     '@width: abc, 1024',
     expect_valid=False, expect_errors=1, error_contains='invalid width')

# --- Sampler/Scheduler validation ---
print('\n--- Sampler/Scheduler Validation ---')

test('Invalid sampler error',
     '@sampler: fake_sampler, euler',
     expect_valid=False, expect_errors=1, error_contains='unknown sampler')

test('Invalid scheduler error',
     '@scheduler: fake_sched, karras',
     expect_valid=False, expect_errors=1, error_contains='unknown scheduler')

# --- Style validation ---
print('\n--- Style Validation ---')

test('Invalid style error',
     '@style: FakeStyle, Fooocus V2',
     expect_valid=False, expect_errors=1, error_contains='unknown style')

test('Multiple styles with one invalid',
     '@style: Fooocus V2|FakeStyle, Cinematic Default',
     expect_valid=False, expect_errors=1, error_contains='unknown style')

test('Empty style value error',
     '@style: "", Cinematic Default',
     expect_valid=False, expect_errors=2)

# --- LoRA weight validation ---
print('\n--- LoRA Weight Validation ---')

test('LoRA weight out of typical range warning',
     '@lora_weight_1: -3.0, 0.8',
     expect_valid=True, expect_errors=0, expect_warnings=1, warning_contains='outside typical range')

test('LoRA weight too high warning',
     '@lora_weight_1: 5.0, 0.8',
     expect_valid=True, expect_errors=0, expect_warnings=1)

test('Invalid LoRA weight error',
     '@lora_weight_1: abc, 0.8',
     expect_valid=False, expect_errors=1, error_contains='invalid LoRA weight value')

test('LoRA index 0 invalid',
     '@lora_weight_0: 0.5, 0.8',
     expect_valid=False, expect_errors=1, error_contains='invalid LoRA index')

# --- Sharpness & Refiner Switch ---
print('\n--- Sharpness & Refiner Switch ---')

test('Sharpness valid',
     '@sharpness: 2.0, 5.0',
     expect_valid=True, expect_errors=0)

test('Sharpness high warning',
     '@sharpness: 15.0, 2.0',
     expect_valid=True, expect_errors=0, expect_warnings=1)

test('Refiner switch > 1 warning',
     '@refiner_switch: 1.5, 0.5',
     expect_valid=True, expect_errors=0, expect_warnings=1, warning_contains='outside valid range')

test('Refiner switch negative warning',
     '@refiner_switch: -0.5, 0.5',
     expect_valid=True, expect_errors=0, expect_warnings=1)

# --- Total tasks calculation ---
print('\n--- Total Tasks Calculation ---')

config = parse_prompt_matrix_config('v1: a, b\nv2: c, d, e')
result = validate_matrix_config(config, image_number=4)
assert result['total_combinations'] == 6
assert result['total_tasks'] == 24, f'Expected 24 tasks, got {result["total_tasks"]}'
print(f'  PASS: total_tasks = combos * image_number ({result["total_combinations"]} * 4 = {result["total_tasks"]})')
passed += 1

# --- Mixed errors and warnings ---
print('\n--- Mixed Scenarios ---')

test('Multiple errors accumulated',
     '@seed: -1, abc\n@lora_weight_1: xyz\nstyle: a, ',
     expect_valid=False, expect_errors=4)

print('\n' + '=' * 60)
print(f'Results: {passed} passed, {failed} failed')
print('=' * 60)

if failed > 0:
    sys.exit(1)
