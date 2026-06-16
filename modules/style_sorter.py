import os
import gradio as gr
import modules.localization as localization
import json
import shutil
from modules.config import sorted_styles_path, _root_dir


def _migrate_old_sorted_styles():
    new_path = sorted_styles_path

    old_paths = []

    project_root_old = os.path.join(_root_dir, 'sorted_styles.json')
    old_paths.append(('project_root', project_root_old))

    cwd_old = os.path.join(os.getcwd(), 'sorted_styles.json')
    if os.path.abspath(cwd_old) != os.path.abspath(project_root_old):
        old_paths.append(('cwd', cwd_old))

    for source_name, old_path in old_paths:
        if os.path.exists(old_path) and not os.path.exists(new_path):
            try:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.move(old_path, new_path)
                print(f'Migrated sorted_styles.json from {source_name} ({old_path}) to new location: {new_path}')
                return
            except Exception as e:
                print(f'Warning: Failed to migrate sorted_styles.json from {old_path} to {new_path}')
                print(e)


_migrate_old_sorted_styles()

all_styles = []


def try_load_sorted_styles(style_names, default_selected):
    global all_styles

    all_styles = style_names

    try:
        if os.path.exists(sorted_styles_path):
            with open(sorted_styles_path, 'rt', encoding='utf-8') as fp:
                sorted_styles = []
                for x in json.load(fp):
                    if x in all_styles:
                        sorted_styles.append(x)
                for x in all_styles:
                    if x not in sorted_styles:
                        sorted_styles.append(x)
                all_styles = sorted_styles
    except Exception as e:
        print('Load style sorting failed.')
        print(e)

    unselected = [y for y in all_styles if y not in default_selected]
    all_styles = default_selected + unselected

    return


def sort_styles(selected):
    global all_styles
    unselected = [y for y in all_styles if y not in selected]
    sorted_styles = selected + unselected
    try:
        os.makedirs(os.path.dirname(sorted_styles_path), exist_ok=True)
        with open(sorted_styles_path, 'wt', encoding='utf-8') as fp:
            json.dump(sorted_styles, fp, indent=4)
    except Exception as e:
        print('Write style sorting failed.')
        print(e)
    all_styles = sorted_styles
    return gr.CheckboxGroup.update(choices=sorted_styles)


def localization_key(x):
    return x + localization.current_translation.get(x, '')


def search_styles(selected, query):
    unselected = [y for y in all_styles if y not in selected]
    matched = [y for y in unselected if query.lower() in localization_key(y).lower()] if len(query.replace(' ', '')) > 0 else []
    unmatched = [y for y in unselected if y not in matched]
    sorted_styles = matched + selected + unmatched
    return gr.CheckboxGroup.update(choices=sorted_styles)
