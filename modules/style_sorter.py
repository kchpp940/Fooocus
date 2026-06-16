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


def _apply_filter(styles_list, filter_mode, selected, query=''):
    selected_set = set(selected)
    special_set = set(_special_styles)

    if filter_mode == 'all' and not query.strip():
        result = styles_list
        selected_only = set()
        return result, selected_only

    if filter_mode == 'favorites':
        favorites = set(style_prefs.get_favorites())
        filter_matches = [s for s in styles_list if s in favorites]

    elif filter_mode == 'recent':
        recent = set(style_prefs.get_recently_used())
        filter_matches = [s for s in styles_list if s in recent]

    else:
        filter_matches = list(styles_list)

    if query and query.strip():
        query_matches = [s for s in filter_matches if query.lower() in localization_key(s).lower()]
        filter_set = set(query_matches)
    else:
        filter_set = set(filter_matches)

    selected_only = selected_set - filter_set - special_set

    keep_set = filter_set | selected_set | special_set
    result = [s for s in styles_list if s in keep_set]

    return result, selected_only


def _apply_grouping(styles_list, group_by, selected, selected_only=None, query=''):
    if selected_only is None:
        selected_only = set()

    if group_by == 'none':
        if query and selected_only:
            special = [s for s in styles_list if s in _special_styles]
            selected = [s for s in styles_list if s in selected_only]
            others = [s for s in styles_list if s not in _special_styles and s not in selected_only]
            return special + selected + others
        return styles_list

    special = [s for s in styles_list if s in _special_styles]
    selected_only_list = [s for s in styles_list if s in selected_only]
    regular = [s for s in styles_list if s not in _special_styles and s not in selected_only]

    result = list(special)

    if selected_only_list:
        result.extend(selected_only_list)

    if group_by == 'source':
        groups = {}
        for s in regular:
            source = sdxl_styles.get_source_label(sdxl_styles.get_style_source(s))
            if source not in groups:
                groups[source] = []
            groups[source].append(s)

        for source in sorted(groups.keys()):
            result.extend(groups[source])
        return result

    elif group_by == 'favorites':
        favorites = set(style_prefs.get_favorites())
        fav_styles = [s for s in regular if s in favorites]
        other_styles = [s for s in regular if s not in favorites]
        return result + fav_styles + other_styles

    return result


def _apply_priority_sort(styles_list, selected, query=''):
    if len(query.replace(' ', '')) > 0:
        matched = [s for s in styles_list if query.lower() in localization_key(s).lower()]
        unmatched = [s for s in styles_list if s not in matched]
        styles_list = matched + unmatched

    selected_list = [s for s in styles_list if s in selected]
    unselected = [s for s in styles_list if s not in selected]

    favorites = set(style_prefs.get_favorites())
    fav_unselected = [s for s in unselected if s in favorites]
    other_unselected = [s for s in unselected if s not in favorites]

    return selected_list + fav_unselected + other_unselected


def _get_group_for_style(style_name, group_by, selected_only):
    if style_name in selected_only:
        return 'Selected Styles'
    if style_name in _special_styles:
        return 'Quick Access'
    if group_by == 'favorites':
        return 'Favorites' if style_prefs.is_favorite(style_name) else 'All Styles'
    if group_by == 'source':
        return sdxl_styles.get_source_label(sdxl_styles.get_style_source(style_name))
    return 'All Styles'


def build_metadata_html(selected_only=None, filter_mode=None, group_by=None):
    source_map = {}
    for style_name in sdxl_styles.style_keys:
        source = sdxl_styles.get_style_source(style_name)
        if source not in source_map:
            source_map[source] = []
        source_map[source].append(style_name)

    current_filter = filter_mode if filter_mode is not None else style_prefs.get_filter()
    current_group = group_by if group_by is not None else style_prefs.get_group_by()

    metadata = {
        'favorites': style_prefs.get_favorites(),
        'recentlyUsed': style_prefs.get_recently_used(),
        'groupBy': current_group,
        'filter': current_filter,
        'sourceMap': source_map,
        'selectedOnly': list(selected_only) if selected_only else []
    }
    return '<script id="style-metadata-data" type="application/json">{}</script>'.format(
        json.dumps(metadata, ensure_ascii=False)
    )


def refresh_style_choices(selected, filter_mode='all', group_by='none', query=''):
    global all_styles

    _validate_style_list(selected, 'input selected')

    styles_list = list(all_styles)

    styles_list, selected_only = _apply_filter(styles_list, filter_mode, selected, query)

    styles_list = _apply_grouping(styles_list, group_by, selected, selected_only, query)
    styles_list = _apply_priority_sort(styles_list, selected, query)

    full_selected = list(selected)

    _validate_style_list(styles_list, 'output choices')
    _validate_style_list(full_selected, 'output value')

    return (
        gr.CheckboxGroup.update(choices=styles_list, value=full_selected),
        build_metadata_html(selected_only, filter_mode, group_by)
    )


def sort_styles(new_visible_order):
    global all_styles
    _validate_style_list(new_visible_order, 'drag sort new order')

    all_style_set = set(all_styles)
    visible_set = set(new_visible_order)

    if visible_set == all_style_set:
        merged_order = list(new_visible_order)
    else:
        visible_iter = iter(new_visible_order)
        merged_order = []
        for s in all_styles:
            if s in visible_set:
                merged_order.append(next(visible_iter))
            else:
                merged_order.append(s)

    _validate_style_list(merged_order, 'drag sort merged result')
    _save_sorted_styles(merged_order)
    all_styles = merged_order
    _rebuild_valid_style_set()

    return refresh_style_choices(new_visible_order)


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
