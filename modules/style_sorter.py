import os
import gradio as gr
import modules.localization as localization
import modules.style_prefs as style_prefs
import modules.sdxl_styles as sdxl_styles
from modules.config import get_user_data_dir
import json


def _get_sorted_styles_path():
    env = os.environ.get('sorted_styles_path')
    if env:
        return env
    return os.path.join(get_user_data_dir(), 'sorted_styles.json')


all_styles = []
_special_styles = ['Fooocus V2', 'Random Style']
_valid_style_set = set()


def _rebuild_valid_style_set():
    global _valid_style_set
    _valid_style_set = set(all_styles) | set(_special_styles)


def _validate_style_list(lst, context='unknown'):
    if not _valid_style_set:
        _rebuild_valid_style_set()
    invalid = [s for s in lst if s not in _valid_style_set]
    if invalid:
        raise ValueError(
            'Style validation failed ({}): {} entries are not real style names: {}'.format(
                context, len(invalid), invalid[:5]
            )
        )


def try_load_sorted_styles(style_names, default_selected):
    global all_styles

    all_styles = style_names
    _rebuild_valid_style_set()
    _validate_style_list(default_selected, 'default_selected')

    sorted_styles_path = _get_sorted_styles_path()
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


def _save_sorted_styles(styles_list):
    sorted_styles_path = _get_sorted_styles_path()
    try:
        os.makedirs(os.path.dirname(sorted_styles_path), exist_ok=True)
        with open(sorted_styles_path, 'wt', encoding='utf-8') as fp:
            json.dump(styles_list, fp, indent=4)
    except Exception as e:
        print('Write style sorting failed.')
        print(e)


def localization_key(x):
    return x + localization.current_translation.get(x, '')


def _apply_filter(styles_list, filter_mode, selected):
    if filter_mode == 'all':
        return styles_list
    elif filter_mode == 'favorites':
        favorites = style_prefs.get_favorites()
        return [s for s in styles_list if s in favorites or s in selected or s in _special_styles]
    elif filter_mode == 'recent':
        recent = style_prefs.get_recently_used()
        return [s for s in styles_list if s in recent or s in selected or s in _special_styles]
    return styles_list


def _apply_grouping(styles_list, group_by, selected):
    if group_by == 'none':
        return styles_list

    special = [s for s in styles_list if s in _special_styles]
    regular = [s for s in styles_list if s not in _special_styles]

    if group_by == 'source':
        groups = {}
        for s in regular:
            source = sdxl_styles.get_source_label(sdxl_styles.get_style_source(s))
            if source not in groups:
                groups[source] = []
            groups[source].append(s)

        result = list(special)
        for source in sorted(groups.keys()):
            result.extend(groups[source])
        return result

    elif group_by == 'favorites':
        favorites = style_prefs.get_favorites()
        fav_styles = [s for s in regular if s in favorites]
        other_styles = [s for s in regular if s not in favorites]
        return list(special) + fav_styles + other_styles

    return styles_list


def _apply_priority_sort(styles_list, selected, query=''):
    if len(query.replace(' ', '')) > 0:
        matched = [s for s in styles_list if query.lower() in localization_key(s).lower()]
        unmatched = [s for s in styles_list if s not in matched]
        styles_list = matched + unmatched

    selected_list = [s for s in styles_list if s in selected]
    unselected = [s for s in styles_list if s not in selected]

    favorites = style_prefs.get_favorites()
    fav_unselected = [s for s in unselected if s in favorites]
    other_unselected = [s for s in unselected if s not in favorites]

    return selected_list + fav_unselected + other_unselected


def build_metadata_html():
    source_map = {}
    for style_name in sdxl_styles.style_keys:
        source = sdxl_styles.get_style_source(style_name)
        if source not in source_map:
            source_map[source] = []
        source_map[source].append(style_name)
    metadata = {
        'favorites': style_prefs.get_favorites(),
        'recentlyUsed': style_prefs.get_recently_used(),
        'groupBy': style_prefs.get_group_by(),
        'filter': style_prefs.get_filter(),
        'sourceMap': source_map
    }
    return '<script id="style-metadata-data" type="application/json">{}</script>'.format(
        json.dumps(metadata, ensure_ascii=False)
    )


def refresh_style_choices(selected, filter_mode='all', group_by='none', query=''):
    global all_styles

    _validate_style_list(selected, 'input selected')

    styles_list = list(all_styles)

    styles_list = _apply_filter(styles_list, filter_mode, selected)
    styles_list = _apply_grouping(styles_list, group_by, selected)
    styles_list = _apply_priority_sort(styles_list, selected, query)

    valid_selected = [s for s in selected if s in styles_list]

    _validate_style_list(styles_list, 'output choices')
    _validate_style_list(valid_selected, 'output value')

    return (
        gr.CheckboxGroup.update(choices=styles_list, value=valid_selected),
        build_metadata_html()
    )


def sort_styles(selected):
    global all_styles
    _validate_style_list(selected, 'drag sort selected')
    unselected = [y for y in all_styles if y not in selected]
    sorted_styles = selected + unselected
    _validate_style_list(sorted_styles, 'drag sort result')
    _save_sorted_styles(sorted_styles)
    all_styles = sorted_styles
    _rebuild_valid_style_set()
    return refresh_style_choices(selected)


def toggle_favorite_and_refresh(style_name, selected, filter_mode, group_by):
    style_prefs.toggle_favorite(style_name)
    return refresh_style_choices(selected, filter_mode, group_by)


def set_filter_and_refresh(selected, filter_mode, current_group):
    style_prefs.set_filter(filter_mode)
    cb_update, meta_html = refresh_style_choices(selected, filter_mode, current_group)
    return cb_update, filter_mode, meta_html


def set_group_and_refresh(selected, group_mode, current_filter):
    group_key = group_mode.lower() if group_mode else 'none'
    style_prefs.set_group_by(group_key)
    cb_update, meta_html = refresh_style_choices(selected, current_filter, group_key)
    return cb_update, group_key, meta_html


def search_styles(selected, query, filter_mode, group_by):
    return refresh_style_choices(selected, filter_mode, group_by, query)


def track_style_usage(style_names):
    for style in style_names:
        if style not in _special_styles:
            if style in _valid_style_set or style in set(all_styles):
                style_prefs.add_to_recently_used(style)
            else:
                print('Warning: skip tracking unknown style name:', style)


def get_favorites():
    return style_prefs.get_favorites()


def get_recently_used():
    return style_prefs.get_recently_used()


def get_group_by():
    return style_prefs.get_group_by()


def get_filter():
    return style_prefs.get_filter()
