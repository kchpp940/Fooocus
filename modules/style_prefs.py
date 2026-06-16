import os
import json
import time
from modules.config import get_user_data_dir


STYLE_PREFS_FILENAME = 'style_preferences.json'
MAX_RECENTLY_USED = 20


def get_style_prefs_path():
    return os.path.join(get_user_data_dir(), STYLE_PREFS_FILENAME)


_default_prefs = {
    'version': 1,
    'favorites': [],
    'recently_used': [],
    'group_by': 'none',
    'filter': 'all'
}


def load_style_prefs():
    prefs_path = get_style_prefs_path()
    try:
        if os.path.exists(prefs_path):
            with open(prefs_path, 'rt', encoding='utf-8') as fp:
                data = json.load(fp)
                merged = _default_prefs.copy()
                merged.update(data)
                return merged
    except Exception as e:
        print('Load style preferences failed.')
        print(e)
    return _default_prefs.copy()


def save_style_prefs(prefs):
    prefs_path = get_style_prefs_path()
    try:
        with open(prefs_path, 'wt', encoding='utf-8') as fp:
            json.dump(prefs, fp, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print('Save style preferences failed.')
        print(e)
        return False


_prefs_cache = None


def get_prefs():
    global _prefs_cache
    if _prefs_cache is None:
        _prefs_cache = load_style_prefs()
    return _prefs_cache


def save_prefs(prefs=None):
    global _prefs_cache
    if prefs is None:
        prefs = _prefs_cache
    if prefs is not None:
        if save_style_prefs(prefs):
            _prefs_cache = prefs


def is_favorite(style_name):
    prefs = get_prefs()
    return style_name in prefs.get('favorites', [])


def toggle_favorite(style_name):
    prefs = get_prefs()
    favorites = prefs.get('favorites', [])
    if style_name in favorites:
        favorites.remove(style_name)
    else:
        favorites.append(style_name)
    prefs['favorites'] = favorites
    save_prefs(prefs)
    return style_name in prefs['favorites']


def get_favorites():
    prefs = get_prefs()
    return prefs.get('favorites', [])


def add_to_recently_used(style_name):
    prefs = get_prefs()
    recently_used = prefs.get('recently_used', [])
    if style_name in recently_used:
        recently_used.remove(style_name)
    recently_used.insert(0, style_name)
    if len(recently_used) > MAX_RECENTLY_USED:
        recently_used = recently_used[:MAX_RECENTLY_USED]
    prefs['recently_used'] = recently_used
    save_prefs(prefs)


def get_recently_used():
    prefs = get_prefs()
    return prefs.get('recently_used', [])


def set_group_by(group_by):
    prefs = get_prefs()
    prefs['group_by'] = group_by
    save_prefs(prefs)


def get_group_by():
    prefs = get_prefs()
    return prefs.get('group_by', 'none')


def set_filter(filter_mode):
    prefs = get_prefs()
    prefs['filter'] = filter_mode
    save_prefs(prefs)


def get_filter():
    prefs = get_prefs()
    return prefs.get('filter', 'all')
