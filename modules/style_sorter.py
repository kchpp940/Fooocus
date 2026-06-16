import os
import gradio as gr
import modules.localization as localization
import modules.style_prefs as style_prefs
import modules.sdxl_styles as sdxl_styles
import json


_sorted_styles_path = os.environ.get(
    'sorted_styles_path',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sorted_styles.json')
)

all_styles = []
_special_styles = ['Fooocus V2', 'Random Style']


def try_load_sorted_styles(style_names, default_selected):
    global all_styles

    all_styles = style_names

    try:
        if os.path.exists(_sorted_styles_path):
            with open(_sorted_styles_path, 'rt', encoding='utf-8') as fp:
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
        with open(_sorted_styles_path, 'wt', encoding='utf-8') as fp:
            json.dump(sorted_styles, fp, indent=4)
    except Exception as e:
        print('Write style sorting failed.')
        print(e)
    all_styles = sorted_styles
    return gr.CheckboxGroup.update(choices=sorted_styles)


def localization_key(x):
    return x + localization.current_translation.get(x, '')


def get_style_display_info(style_name):
    source_file = sdxl_styles.get_style_source(style_name)
    return {
        'name': style_name,
        'source': source_file,
        'source_label': sdxl_styles.get_source_label(source_file),
        'is_favorite': style_prefs.is_favorite(style_name),
        'is_recent': style_name in style_prefs.get_recently_used()
    }


def toggle_favorite(style_name):
    result = style_prefs.toggle_favorite(style_name)
    return result


def is_favorite(style_name):
    return style_prefs.is_favorite(style_name)


def get_favorites():
    return style_prefs.get_favorites()


def get_recently_used():
    return style_prefs.get_recently_used()


def track_style_usage(style_names):
    for style in style_names:
        if style not in _special_styles:
            style_prefs.add_to_recently_used(style)


def set_group_by(group_by):
    style_prefs.set_group_by(group_by)


def get_group_by():
    return style_prefs.get_group_by()


def set_filter(filter_mode):
    style_prefs.set_filter(filter_mode)


def get_filter():
    return style_prefs.get_filter()


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


def search_styles(selected, query, filter_mode='all', group_by='none'):
    global all_styles

    styles_list = list(all_styles)

    styles_list = _apply_filter(styles_list, filter_mode, selected)
    styles_list = _apply_grouping(styles_list, group_by, selected)
    styles_list = _apply_priority_sort(styles_list, selected, query)

    return gr.CheckboxGroup.update(choices=styles_list)


def refresh_styles_display(selected, filter_mode='all', group_by='none'):
    return search_styles(selected, '', filter_mode, group_by)


def get_style_groups(selected, filter_mode='all', group_by='none'):
    styles_list = list(all_styles)
    styles_list = _apply_filter(styles_list, filter_mode, selected)
    styles_list = _apply_grouping(styles_list, group_by, selected)
    styles_list = _apply_priority_sort(styles_list, selected, '')

    groups = []
    if group_by == 'source':
        current_source = None
        for s in styles_list:
            if s in _special_styles:
                source = 'Quick Access'
            else:
                source = sdxl_styles.get_source_label(sdxl_styles.get_style_source(s))
            if source != current_source:
                groups.append({'type': 'header', 'label': source})
                current_source = source
            groups.append({'type': 'style', 'name': s})
    elif group_by == 'favorites':
        favorites = style_prefs.get_favorites()
        current_group = None
        for s in styles_list:
            if s in _special_styles:
                group = 'Quick Access'
            elif s in favorites:
                group = 'Favorites'
            else:
                group = 'All Styles'
            if group != current_group:
                groups.append({'type': 'header', 'label': group})
                current_group = group
            groups.append({'type': 'style', 'name': s})
    else:
        for s in styles_list:
            groups.append({'type': 'style', 'name': s})

    return groups
