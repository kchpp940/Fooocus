import os
import ldm_patched.modules.args_parser as args_parser

from modules.config_schema import ConfigSchema, build_fooocus_schema

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_schema: ConfigSchema = build_fooocus_schema(_root_dir)

for field in config_schema.get_cli_fields():
    cli_arg = f"--{field.cli_arg}"
    if field.cli_action == 'store_true':
        args_parser.parser.add_argument(
            cli_arg,
            action='store_true',
            help=field.description,
            default=field.default_value
        )
    elif field.cli_arg == 'rebuild-hash-cache':
        args_parser.parser.add_argument(
            cli_arg,
            help=field.description,
            type=int,
            nargs="?",
            metavar="CPU_NUM_THREADS",
            const=-1,
            default=field.default_value
        )
    else:
        args_parser.parser.add_argument(
            cli_arg,
            type=field.expected_type if field.expected_type in (str, int, float) else str,
            default=field.default_value,
            help=field.description
        )

args_parser.parser.set_defaults(
    disable_cuda_malloc=True,
    in_browser=True,
    port=None
)

args_parser.args = args_parser.parser.parse_args()

args_parser.args.always_offload_from_vram = not args_parser.args.disable_offload_from_vram

if args_parser.args.disable_analytics:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if args_parser.args.disable_in_browser:
    args_parser.args.in_browser = False

args = args_parser.args


def sync_cli_args_to_config_result(config_result):
    issues = config_schema.load_cli_args(config_result, args)

    if args.output_path:
        from modules.config_schema import ConfigSource
        issues.extend(config_schema.apply_value(
            config_result, 'path_outputs', args.output_path,
            ConfigSource.CLI_ARGUMENT, 'cli:output-path'
        ))

    if args.temp_path:
        from modules.config_schema import ConfigSource
        issues.extend(config_schema.apply_value(
            config_result, 'temp_path', args.temp_path,
            ConfigSource.CLI_ARGUMENT, 'cli:temp-path'
        ))

    return issues
