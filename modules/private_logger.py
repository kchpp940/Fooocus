import os
import args_manager
import modules.config
import json
import urllib.parse

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from modules.flags import OutputFormat
from modules.meta_parser import MetadataParser, get_exif
from modules.util import generate_temp_filename
import modules.diagnostics as diagnostics
from modules.diagnostics import (
    DiagnosticStage, DiagnosticErrorCategory, get_current_context,
    log_info, log_error, log_warning,
)

log_cache = {}


def get_current_html_path(output_format=None):
    output_format = output_format if output_format else modules.config.default_output_format
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.config.path_outputs,
                                                                         extension=output_format)
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')
    return html_name


def log(img, metadata, metadata_parser: MetadataParser | None = None, output_format=None, task=None, persist_image=True) -> str:
    ctx = get_current_context()
    path_outputs = modules.config.temp_path if args_manager.args.disable_image_log or not persist_image else modules.config.path_outputs
    output_format = output_format if output_format else modules.config.default_output_format
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=path_outputs, extension=output_format)
    os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)

    try:
        parsed_parameters = metadata_parser.to_string(metadata.copy()) if metadata_parser is not None else ''
        image = Image.fromarray(img)

        if output_format == OutputFormat.PNG.value:
            if parsed_parameters != '':
                pnginfo = PngInfo()
                pnginfo.add_text('parameters', parsed_parameters)
                pnginfo.add_text('fooocus_scheme', metadata_parser.get_scheme().value)
            else:
                pnginfo = None
            image.save(local_temp_filename, pnginfo=pnginfo)
        elif output_format == OutputFormat.JPEG.value:
            image.save(local_temp_filename, quality=95, optimize=True, progressive=True, exif=get_exif(parsed_parameters, metadata_parser.get_scheme().value) if metadata_parser else Image.Exif())
        elif output_format == OutputFormat.WEBP.value:
            image.save(local_temp_filename, quality=95, lossless=False, exif=get_exif(parsed_parameters, metadata_parser.get_scheme().value) if metadata_parser else Image.Exif())
        else:
            image.save(local_temp_filename)

        if ctx:
            log_info(
                DiagnosticStage.IMAGE_SAVE,
                f"图像保存成功: {only_name}",
                ctx=ctx,
                extra_data={
                    "filename": only_name,
                    "format": output_format,
                    "size": os.path.getsize(local_temp_filename) if os.path.exists(local_temp_filename) else None,
                    "has_metadata": parsed_parameters != '',
                }
            )

    except Exception as e:
        log_error(
            DiagnosticStage.IMAGE_SAVE,
            "图像保存失败",
            exception=e,
            category=DiagnosticErrorCategory.IMAGE_SAVE_FAILED,
            ctx=ctx,
            extra_data={
                "output_format": output_format,
                "target_path": local_temp_filename,
                "has_metadata_parser": metadata_parser is not None,
            }
        )
        raise diagnostics.DiagnosticsError(
            human_message=f"图像保存失败: {os.path.basename(local_temp_filename)}",
            category=DiagnosticErrorCategory.IMAGE_SAVE_FAILED,
            stage=DiagnosticStage.IMAGE_SAVE,
            ctx=ctx,
            cause=e,
        )

    if args_manager.args.disable_image_log:
        return local_temp_filename

    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')

    css_styles = (
        "<style>"
        "body { background-color: #121212; color: #E0E0E0; } "
        "a { color: #BB86FC; } "
        ".metadata { border-collapse: collapse; width: 100%; } "
        ".metadata .label { width: 15%; } "
        ".metadata .value { width: 85%; font-weight: bold; } "
        ".metadata th, .metadata td { border: 1px solid #4d4d4d; padding: 4px; } "
        ".image-container img { height: auto; max-width: 512px; display: block; padding-right:10px; } "
        ".image-container div { text-align: center; padding: 4px; } "
        "hr { border-color: gray; } "
        "button { background-color: black; color: white; border: 1px solid grey; border-radius: 5px; padding: 5px 10px; text-align: center; display: inline-block; font-size: 16px; cursor: pointer; }"
        "button:hover {background-color: grey; color: black;}"
        "</style>"
    )

    js = (
        """<script>
        function to_clipboard(txt) { 
        txt = decodeURIComponent(txt);
        if (navigator.clipboard && navigator.permissions) {
            navigator.clipboard.writeText(txt)
        } else {
            const textArea = document.createElement('textArea')
            textArea.value = txt
            textArea.style.width = 0
            textArea.style.position = 'fixed'
            textArea.style.left = '-999px'
            textArea.style.top = '10px'
            textArea.setAttribute('readonly', 'readonly')
            document.body.appendChild(textArea)

            textArea.select()
            document.execCommand('copy')
            document.body.removeChild(textArea)
        }
        alert('Copied to Clipboard!\\nPaste to prompt area to load parameters.\\nCurrent clipboard content is:\\n\\n' + txt);
        }
        </script>"""
    )

    begin_part = f"<!DOCTYPE html><html><head><title>Fooocus Log {date_string}</title>{css_styles}</head><body>{js}<p>Fooocus Log {date_string} (private)</p>\n<p>Metadata is embedded if enabled in the config or developer debug mode. You can find the information for each image in line Metadata Scheme.</p><!--fooocus-log-split-->\n\n"
    end_part = f'\n<!--fooocus-log-split--></body></html>'

    middle_part = log_cache.get(html_name, "")

    if middle_part == "":
        if os.path.exists(html_name):
            existing_split = open(html_name, 'r', encoding='utf-8').read().split('<!--fooocus-log-split-->')
            if len(existing_split) == 3:
                middle_part = existing_split[1]
            else:
                middle_part = existing_split[0]

    div_name = only_name.replace('.', '_')
    item = f"<div id=\"{div_name}\" class=\"image-container\"><hr><table><tr>\n"
    item += f"<td><a href=\"{only_name}\" target=\"_blank\"><img src='{only_name}' onerror=\"this.closest('.image-container').style.display='none';\" loading='lazy'/></a><div>{only_name}</div></td>"
    item += "<td><table class='metadata'>"
    for label, key, value in metadata:
        value_txt = str(value).replace('\n', ' </br> ')
        item += f"<tr><td class='label'>{label}</td><td class='value'>{value_txt}</td></tr>\n"

    if task is not None and 'positive' in task and 'negative' in task:
        full_prompt_details = f"""<details><summary>Positive</summary>{', '.join(task['positive'])}</details>
        <details><summary>Negative</summary>{', '.join(task['negative'])}</details>"""
        item += f"<tr><td class='label'>Full raw prompt</td><td class='value'>{full_prompt_details}</td></tr>\n"

    item += "</table>"

    js_txt = urllib.parse.quote(json.dumps({k: v for _, k, v, in metadata}, indent=0), safe='')
    item += f"</br><button onclick=\"to_clipboard('{js_txt}')\">Copy to Clipboard</button>"

    item += "</td>"
    item += "</tr></table></div>\n\n"

    middle_part = item + middle_part

    try:
        with open(html_name, 'w', encoding='utf-8') as f:
            f.write(begin_part + middle_part + end_part)

        log_info(
            DiagnosticStage.IMAGE_SAVE,
            f"HTML 日志已更新: {os.path.basename(html_name)}",
            ctx=ctx,
            extra_data={"html_log_path": html_name},
        )

        print(f'Image generated with private log at: {html_name}')

        log_cache[html_name] = middle_part
    except Exception as e:
        log_warning(
            DiagnosticStage.IMAGE_SAVE,
            "HTML 日志写入失败，图像已保存但日志未更新",
            ctx=ctx,
            extra_data={
                "html_log_path": html_name,
                "error": str(e),
            },
        )

    return local_temp_filename
