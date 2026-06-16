// based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/script.js
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

/**
 * Get the currently selected top-level UI tab button (e.g. the button that says "Extras").
 */
function get_uiCurrentTab() {
    return gradioApp().querySelector('#tabs > .tab-nav > button.selected');
}

/**
 * Get the first currently visible top-level UI tab content (e.g. the div hosting the "txt2img" UI).
 */
function get_uiCurrentTabContent() {
    return gradioApp().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])');
}

var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];
var uiLoadedCallbacks = [];
var uiTabChangeCallbacks = [];
var optionsChangedCallbacks = [];
var uiAfterUpdateTimeout = null;
var uiCurrentTab = null;

/**
 * Register callback to be called at each UI update.
 * The callback receives an array of MutationRecords as an argument.
 */
function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called soon after UI updates.
 * The callback receives no arguments.
 *
 * This is preferred over `onUiUpdate` if you don't need
 * access to the MutationRecords, as your function will
 * not be called quite as often.
 */
function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI is loaded.
 * The callback receives no arguments.
 */
function onUiLoaded(callback) {
    uiLoadedCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI tab is changed.
 * The callback receives no arguments.
 */
function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}

/**
 * Register callback to be called when the options are changed.
 * The callback receives no arguments.
 * @param callback
 */
function onOptionsChanged(callback) {
    optionsChangedCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * The callbacks are executed after a short while, unless another call to this function
 * is made before that time. IOW, the callbacks are executed only once, even
 * when there are multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 200);
}

var executedOnLoaded = false;

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m) {
        if (!executedOnLoaded && gradioApp().querySelector('#generate_button')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, m);
        scheduleAfterUiUpdateCallbacks();
        const newTab = get_uiCurrentTab();
        if (newTab && (newTab !== uiCurrentTab)) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
    initStylePreviewOverlay();
});

var onAppend = function(elem, f) {
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(m) {
            if (m.addedNodes.length) {
                f(m.addedNodes);
            }
        });
    });
    observer.observe(elem, {childList: true});
}

function addObserverIfDesiredNodeAvailable(querySelector, callback) {
    var elem = document.querySelector(querySelector);
    if (!elem) {
        window.setTimeout(() => addObserverIfDesiredNodeAvailable(querySelector, callback), 1000);
        return;
    }

    onAppend(elem, callback);
}

/**
 * Show reset button on toast "Connection errored out."
 */
addObserverIfDesiredNodeAvailable(".toast-wrap", function(added) {
    added.forEach(function(element) {
         if (element.innerText.includes("Connection errored out.")) {
             window.setTimeout(function() {
                document.getElementById("reset_button").classList.remove("hidden");
                document.getElementById("generate_button").classList.add("hidden");
                document.getElementById("skip_button").classList.add("hidden");
                document.getElementById("stop_button").classList.add("hidden");
            });
         }
    });
});

/**
 * Add a ctrl+enter as a shortcut to start a generation
 */
document.addEventListener('keydown', function(e) {
    const isModifierKey = (e.metaKey || e.ctrlKey || e.altKey);
    const isEnterKey = (e.key == "Enter" || e.keyCode == 13);

    if(isModifierKey && isEnterKey) {
        const generateButton = gradioApp().querySelector('button:not(.hidden)[id=generate_button]');
        if (generateButton) {
            generateButton.click();
            e.preventDefault();
            return;
        }

        const stopButton = gradioApp().querySelector('button:not(.hidden)[id=stop_button]')
        if(stopButton) {
            stopButton.click();
            e.preventDefault();
            return;
        }
    }
});

function initStylePreviewOverlay() {
    let overlayVisible = false;
    const samplesPath = document.querySelector("meta[name='samples-path']").getAttribute("content")
    const overlay = document.createElement('div');
    const tooltip = document.createElement('div');
    tooltip.className = 'preview-tooltip';
    overlay.appendChild(tooltip);
    overlay.id = 'stylePreviewOverlay';
    document.body.appendChild(overlay);
    document.addEventListener('mouseover', function (e) {
        const label = e.target.closest('.style_selections label');
        if (!label) return;
        label.removeEventListener("mouseout", onMouseLeave);
        label.addEventListener("mouseout", onMouseLeave);
        overlayVisible = true;
        overlay.style.opacity = "1";
        const originalText = label.querySelector("span").getAttribute("data-original-text");
        const name = originalText || label.querySelector("span").textContent;
        overlay.style.backgroundImage = `url("${samplesPath.replace(
            "fooocus_v2",
            name.toLowerCase().replaceAll(" ", "_")
        ).replaceAll("\\", "\\\\")}")`;

        tooltip.textContent = name;

        function onMouseLeave() {
            overlayVisible = false;
            overlay.style.opacity = "0";
            overlay.style.backgroundImage = "";
            label.removeEventListener("mouseout", onMouseLeave);
        }
    });
    document.addEventListener('mousemove', function (e) {
        if (!overlayVisible) return;
        overlay.style.left = `${e.clientX}px`;
        overlay.style.top = `${e.clientY}px`;
        overlay.className = e.clientY > window.innerHeight / 2 ? "lower-half" : "upper-half";
    });
}

/**
 * checks that a UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
    if (el === document) {
        return true;
    }

    const computedStyle = getComputedStyle(el);
    const isVisible = computedStyle.display !== 'none';

    if (!isVisible) return false;
    return uiElementIsVisible(el.parentNode);
}

function uiElementInSight(el) {
    const clRect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;

    return isOnScreen;
}

function playNotification() {
    gradioApp().querySelector('#audio_notification audio')?.play();
}

function set_theme(theme) {
    var gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function htmlDecode(input) {
  var doc = new DOMParser().parseFromString(input, "text/html");
  return doc.documentElement.textContent;
}

var styleFavoritesData = {
    favorites: [],
    recentlyUsed: [],
    groupBy: 'none',
    filter: 'all',
    sourceMap: {}
};

function updateStyleFavoritesData(data) {
    if (data) {
        Object.assign(styleFavoritesData, data);
    }
}

function getStyleNameFromLabel(label) {
    var span = label.querySelector('span');
    if (!span) return null;
    var originalText = span.getAttribute('data-original-text');
    return originalText || span.textContent.trim();
}

function addFavoriteStarsToStyles() {
    var container = document.querySelector('.style_selections .wrap[data-testid="checkbox-group"]');
    if (!container) return;

    container.querySelectorAll('.style_favorite_star').forEach(function(star) {
        star.remove();
    });

    container.querySelectorAll('label').forEach(function(label) {
        if (label.querySelector('.style_favorite_star')) return;

        var styleName = getStyleNameFromLabel(label);
        if (!styleName) return;

        var star = document.createElement('span');
        star.className = 'style_favorite_star';
        star.innerHTML = '\u2606';
        star.title = 'Add to Favorites';

        if (styleFavoritesData.favorites.indexOf(styleName) > -1) {
            star.classList.add('is_favorite');
            star.innerHTML = '\u2605';
            star.title = 'Remove from Favorites';
        }

        star.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleFavoriteStyle(styleName);
        });

        label.insertBefore(star, label.firstChild);
    });
}

function toggleFavoriteStyle(styleName) {
    var receiver = document.getElementById('gradio_receiver_favorite_toggle');
    if (receiver) {
        receiver.value = styleName;
        var event = new Event('input', { bubbles: true });
        receiver.dispatchEvent(event);
    }
}

function setActiveStyleFilter(filter) {
    styleFavoritesData.filter = filter;

    document.querySelectorAll('.style_filter_btn').forEach(function(btn) {
        btn.classList.remove('style_filter_active');
    });

    var filterLabels = {
        'all': 'All',
        'favorites': 'Favorites',
        'recent': 'Recent'
    };

    document.querySelectorAll('.style_filter_btn').forEach(function(btn) {
        var btnText = btn.textContent.trim();
        if (btnText === filterLabels[filter]) {
            btn.classList.add('style_filter_active');
        }
    });
}

function setActiveGroupBy(groupBy) {
    var groupKey = groupBy.toLowerCase();
    if (groupKey === 'none' || groupKey === 'source' || groupKey === 'favorites') {
        styleFavoritesData.groupBy = groupKey;
    } else {
        styleFavoritesData.groupBy = 'none';
    }
}

function addStyleGroupHeaders() {
    var container = document.querySelector('.style_selections .wrap[data-testid="checkbox-group"]');
    if (!container) return;

    container.querySelectorAll('.style_group_header').forEach(function(h) {
        h.remove();
    });

    var hasSelectedOnly = styleFavoritesData.selectedOnly && styleFavoritesData.selectedOnly.length > 0;
    if (styleFavoritesData.groupBy === 'none' && !hasSelectedOnly) {
        return;
    }

    var labels = container.querySelectorAll('label');
    var currentGroup = null;

    labels.forEach(function(label) {
        var styleName = getStyleNameFromLabel(label);
        if (!styleName) return;

        var group = getStyleGroup(styleName);

        if (group !== currentGroup) {
            var header = document.createElement('div');
            header.className = 'style_group_header';
            if (styleFavoritesData.selectedOnly && styleFavoritesData.selectedOnly.indexOf(styleName) > -1) {
                header.classList.add('selected_only_group');
            }
            header.textContent = group;
            container.insertBefore(header, label);
            currentGroup = group;
        }
    });
}

function getStyleGroup(styleName) {
    if (styleFavoritesData.selectedOnly && styleFavoritesData.selectedOnly.indexOf(styleName) > -1) {
        return 'Selected Styles';
    }

    if (['Fooocus V2', 'Random Style'].indexOf(styleName) > -1) {
        return 'Quick Access';
    }

    if (styleFavoritesData.groupBy === 'favorites') {
        return styleFavoritesData.favorites.indexOf(styleName) > -1 ? 'Favorites' : 'All Styles';
    }

    if (styleFavoritesData.groupBy === 'source') {
        var sourceLabels = {
            'sdxl_styles_fooocus.json': 'Fooocus',
            'sdxl_styles_sai.json': 'SAI',
            'sdxl_styles_mre.json': 'MRE',
            'sdxl_styles_twri.json': 'Twri',
            'sdxl_styles_diva.json': 'Diva',
            'sdxl_styles_marc_k3nt3l.json': 'Marc K3nt3l'
        };

        for (var source in styleFavoritesData.sourceMap || {}) {
            if (styleFavoritesData.sourceMap[source].indexOf(styleName) > -1) {
                return sourceLabels[source] || source;
            }
        }
        return 'Other';
    }

    return 'All Styles';
}

function initializeStyleManagement() {
    loadStyleMetadata();
    enhanceStyleSelections();
}

function loadStyleMetadata() {
    var metaScript = document.getElementById('style-metadata-data');
    if (metaScript) {
        try {
            var data = JSON.parse(metaScript.textContent);
            updateStyleFavoritesData(data);
        } catch (e) {
            console.log('Could not parse style metadata:', e);
        }
    }
}

function enhanceStyleSelections() {
    addFavoriteStarsToStyles();
    addStyleGroupHeaders();
}

var originalRefreshStyleLocalization = window.refresh_style_localization;
window.refresh_style_localization = function() {
    if (originalRefreshStyleLocalization) {
        originalRefreshStyleLocalization();
    }
    setTimeout(enhanceStyleSelections, 50);
};