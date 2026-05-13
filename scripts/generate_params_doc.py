#!/usr/bin/env python3
"""
Generate parameter documentation for CommonParams and its sub-classes.

Extracts Python-exposed properties (name, type, docstring) from the Cython
.pyx wrapper, cross-references them with default values and comments from the
C++ common.h header, and produces a Markdown reference document.

Usage:
    python scripts/generate_params_doc.py                    # write to params_reference.md
    python scripts/generate_params_doc.py -o docs/ref.md     # write to a specific file
    python scripts/generate_params_doc.py -o -               # write to stdout
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYX_FILE = PROJECT_ROOT / "src" / "xllamacpp" / "xllamacpp.pyx"
COMMON_H = PROJECT_ROOT / "thirdparty" / "llama.cpp" / "common" / "common.h"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PropertyInfo:
    name: str
    python_type: str = ""
    docstring: str = ""
    has_setter: bool = False


@dataclass
class ClassInfo:
    name: str
    properties: list[PropertyInfo] = field(default_factory=list)
    description: str = ""


@dataclass
class CppFieldInfo:
    """Field information extracted from common.h."""
    name: str
    cpp_type: str = ""
    default: str = ""
    comment: str = ""


# ---------------------------------------------------------------------------
# Parse .pyx file
# ---------------------------------------------------------------------------

# Classes we care about
TARGET_CLASSES = [
    "CommonParams",
    "CommonParamsSampling",
    "CommonParamsModel",
    "CommonParamsSpeculative",
    "CommonParamsVocoder",
    "CommonParamsDiffusion",
    "CpuParams",
    "CommonAdapterLoraInfo",
]

# Class descriptions (English)
CLASS_DESCRIPTIONS = {
    "CommonParams": "The central configuration object. Controls model loading, inference, sampling, server behavior, and more. Create via `xlc.CommonParams()`.",
    "CommonParamsSampling": "Sampling parameters that control token generation strategy. Access via `params.sampling`.",
    "CommonParamsModel": "Model path and source parameters. Access via `params.model`.",
    "CommonParamsSpeculative": "Speculative decoding parameters. Access via `params.speculative`.",
    "CommonParamsVocoder": "Text-to-speech (vocoder) parameters. Access via `params.vocoder`.",
    "CommonParamsDiffusion": "Diffusion model parameters. Access via `params.diffusion`.",
    "CpuParams": "CPU threading and scheduling parameters. Access via `params.cpuparams` or `params.cpuparams_batch`.",
    "CommonAdapterLoraInfo": "LoRA adapter info. Create via `xlc.CommonAdapterLoraInfo(path, scale)`.",
}


def _extract_first_docstring_line(raw: str) -> str:
    """Return only the first meaningful sentence of a multi-line docstring.

    Multi-line docstrings in the .pyx file often contain internal C++ type
    notes on subsequent lines (e.g. ``std_vector[llama_logit_bias] logit_bias``).
    We keep only the first non-empty line as the user-facing description.
    """
    for line in raw.split("\n"):
        line = line.strip()
        if line:
            return line
    return raw.strip()


def parse_pyx(pyx_path: Path) -> dict[str, ClassInfo]:
    """Parse a .pyx file and extract class / property information."""
    text = pyx_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    classes: dict[str, ClassInfo] = {}
    current_class: ClassInfo | None = None
    current_prop: PropertyInfo | None = None
    in_property = False
    in_setter = False
    in_docstring = False
    docstring_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Detect class definition
        class_match = re.match(r"^cdef class (\w+)", stripped)
        if class_match:
            cls_name = class_match.group(1)
            if cls_name in TARGET_CLASSES:
                current_class = ClassInfo(name=cls_name)
                classes[cls_name] = current_class
                current_prop = None
                in_property = False
                in_setter = False
            else:
                current_class = None
            continue

        if current_class is None:
            continue

        # Detect @property decorator
        if stripped == "@property":
            in_property = True
            in_setter = False
            continue

        # Detect setter decorator
        setter_match = re.match(r"^@(\w+)\.setter", stripped)
        if setter_match:
            in_setter = True
            in_property = False
            # Mark the corresponding property as having a setter
            prop_name = setter_match.group(1)
            for p in current_class.properties:
                if p.name == prop_name:
                    p.has_setter = True
                    break
            continue

        # After @property, parse the def line
        if in_property:
            def_match = re.match(r"^\s*def\s+(\w+)\s*\(self\)\s*(?:->\s*(.+?))?\s*:", stripped)
            if def_match:
                prop_name = def_match.group(1)
                prop_type = (def_match.group(2) or "").strip()
                current_prop = PropertyInfo(name=prop_name, python_type=prop_type)
                current_class.properties.append(current_prop)
                in_property = False
                in_docstring = False
                docstring_lines = []
                continue

        # Parse docstring (immediately after the property def)
        if current_prop and not in_setter:
            if stripped.startswith('"""') and not in_docstring:
                # Might be a single-line docstring: """some text"""
                if stripped.endswith('"""') and len(stripped) > 3:
                    doc = stripped[3:-3].strip()
                    current_prop.docstring = doc
                    current_prop = None
                else:
                    in_docstring = True
                    doc_start = stripped[3:]  # remove leading """
                    if doc_start.strip():
                        docstring_lines.append(doc_start.strip())
                continue
            elif in_docstring:
                if '"""' in stripped:
                    doc_end = stripped.replace('"""', "").strip()
                    if doc_end:
                        docstring_lines.append(doc_end)
                    # Use only the first meaningful line to avoid leaking
                    # internal C++ type notes into the description.
                    full = "\n".join(docstring_lines)
                    current_prop.docstring = _extract_first_docstring_line(full)
                    in_docstring = False
                    current_prop = None
                else:
                    docstring_lines.append(stripped)
                continue

            # If we hit a return or cdef statement, stop tracking this property
            if stripped.startswith("return ") or stripped.startswith("cdef "):
                current_prop = None

    return classes


# ---------------------------------------------------------------------------
# Parse common.h
# ---------------------------------------------------------------------------

# Pre-compiled pattern for stripping string literals
_RE_STRING_LITERAL = re.compile(r'"[^"]*"')


def parse_common_h(h_path: Path) -> dict[str, dict[str, CppFieldInfo]]:
    """Parse common.h and extract struct field information.

    Returns ``{python_class_name: {field_name: CppFieldInfo}}``.
    """
    text = h_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    structs: dict[str, dict[str, CppFieldInfo]] = {}

    # Map C++ struct names to Python class names
    struct_to_class = {
        "common_params": "CommonParams",
        "common_params_sampling": "CommonParamsSampling",
        "common_params_model": "CommonParamsModel",
        "common_params_speculative": "CommonParamsSpeculative",
        "common_params_vocoder": "CommonParamsVocoder",
        "common_params_diffusion": "CommonParamsDiffusion",
        "cpu_params": "CpuParams",
        "common_adapter_lora_info": "CommonAdapterLoraInfo",
    }

    current_struct: str | None = None
    current_fields: dict[str, CppFieldInfo] = {}
    brace_depth = 0

    def _structural_brace_delta(s: str) -> int:
        """Count brace depth change, ignoring braces inside initializers and strings."""
        # Only count structural braces; strip everything after '='
        eq_pos = s.find("=")
        if eq_pos >= 0:
            s = s[:eq_pos]
        # Remove string literals
        s = _RE_STRING_LITERAL.sub("", s)
        return s.count("{") - s.count("}")

    for line in lines:
        stripped = line.strip()

        # Detect struct definition (must contain '{' to be a definition)
        struct_match = re.match(r"^struct\s+(\w+)\s*\{", stripped)
        if struct_match:
            name = struct_match.group(1)
            if name in struct_to_class:
                current_struct = struct_to_class[name]
                current_fields = {}
                structs[current_struct] = current_fields
                brace_depth = _structural_brace_delta(stripped)
                continue

        if current_struct is None:
            continue

        brace_depth += _structural_brace_delta(stripped)
        if brace_depth <= 0:
            current_struct = None
            continue

        # Skip blank lines, pure comment lines, and using declarations
        if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
            continue
        if stripped.startswith("using "):
            continue

        # Must contain a semicolon to be a field declaration
        if ";" not in stripped:
            continue

        # Skip method declarations
        if re.match(r"^(bool|void|std::string|int|float|size_t|auto)\s+\w+\s*\(", stripped):
            continue
        if stripped.startswith("~"):
            continue

        # Extract trailing comment (after the semicolon)
        comment = ""
        semi_pos = stripped.find(";")
        if semi_pos < 0:
            continue
        after_semi = stripped[semi_pos + 1:].strip()
        if after_semi.startswith("//"):
            comment = after_semi[2:].strip()
            comment = re.sub(r"\s*//\s*NOLINT\s*$", "", comment)

        before_semi = stripped[:semi_pos]

        # Try to parse: [qualifiers] type name [= default];
        field_match = re.match(
            r"""
            ^
            (?:(?:enum|struct)\s+)?                  # optional enum/struct keyword
            (                                        # type group
                (?:const\s+)?                        # optional const
                [\w:]+                               # base type (e.g. int32_t, std::string, bool)
                (?:\s*<[^>]*(?:<[^>]*>[^>]*)?>)?     # optional template params (nested)
                (?:\s*\*)?                            # optional pointer
            )
            \s+
            (\w+)                                    # field name
            (?:\[\w*\])?                             # optional array brackets
            (?:\s*=\s*(.+?))?                        # optional default value (lazy)
            \s*$                                     # end
            """,
            before_semi.strip(),
            re.VERBOSE,
        )
        if field_match:
            cpp_type = field_match.group(1).strip()
            fname = field_match.group(2).strip()
            default = (field_match.group(3) or "").strip()

            # Clean up the default value
            default = default.rstrip(",").strip()
            default = re.sub(r"\s*//\s*NOLINT.*$", "", default).strip()
            default = re.sub(r"^NOLINT.*$", "", default).strip()
            # Remove C++ float suffix (e.g. 0.5f -> 0.5)
            default = re.sub(r"(\d+\.\d+)f\b", r"\1", default)
            # Clean up comment
            comment = re.sub(r"\s*//\s*NOLINT.*$", "", comment).strip()
            comment = re.sub(r"^NOLINT.*$", "", comment).strip()

            current_fields[fname] = CppFieldInfo(
                name=fname,
                cpp_type=cpp_type,
                default=default,
                comment=comment,
            )

    return structs


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

# Output order (alphabetical)
CLASS_ORDER = [
    "CommonAdapterLoraInfo",
    "CommonParams",
    "CommonParamsDiffusion",
    "CommonParamsModel",
    "CommonParamsSampling",
    "CommonParamsSpeculative",
    "CommonParamsVocoder",
    "CpuParams",
]

# Property groups (CommonParams only, alphabetical within each group)
COMMON_PARAMS_GROUPS = [
    ("Attention / KV Cache", [
        "pooling_type", "attention_type", "flash_attn_type",
        "cache_type_k", "cache_type_v",
        "ctx_shift", "swa_full", "kv_unified",
    ]),
    ("Batched Bench", [
        "is_pp_shared", "is_tg_separate", "n_pp", "n_tg", "n_pl",
    ]),
    ("Behavior Flags", [
        "usage", "use_color", "special", "interactive", "single_turn",
        "prompt_cache_all", "prompt_cache_ro",
        "escape", "multiline_input", "simple_io",
        "cont_batching", "no_perf", "show_timings",
        "use_mmap", "use_direct_io", "use_mlock",
        "verbose_prompt", "display_prompt", "warmup", "check_tensors",
        "offline",
    ]),
    ("CPU", [
        "cpuparams", "cpuparams_batch", "numa",
    ]),
    ("CVector Generator", [
        "n_pca_batch", "n_pca_iterations",
    ]),
    ("Control Vector", [
        "control_vector_layer_start", "control_vector_layer_end",
    ]),
    ("Embedding", [
        "embedding", "embd_normalize", "embd_out", "embd_sep", "cls_sep",
    ]),
    ("GPU / Offloading", [
        "n_gpu_layers", "main_gpu", "tensor_split", "split_mode",
        "fit_params", "fit_params_print", "fit_params_min_ctx", "fit_params_target",
        "no_kv_offload", "no_op_offload", "no_extra_bufts", "no_host",
    ]),
    ("IMatrix", [
        "n_out_freq", "n_save_freq", "i_chunk", "imat_dat",
        "process_output", "compute_ppl", "show_statistics", "parse_special",
    ]),
    ("Inference", [
        "n_predict", "n_ctx", "n_batch", "n_ubatch", "n_keep", "n_chunks",
        "n_parallel", "n_sequences", "grp_attn_n", "grp_attn_w", "n_print",
    ]),
    ("LoRA", [
        "lora_init_without_apply", "lora_adapters",
    ]),
    ("Model", [
        "model", "model_alias", "model_tags", "hf_token",
    ]),
    ("Multimodal", [
        "mmproj", "mmproj_use_gpu", "no_mmproj", "image", "image_min_tokens", "image_max_tokens",
    ]),
    ("Output / Logging", [
        "logits_file", "logits_output_dir", "save_logits", "tensor_filter",
        "verbosity", "out_file",
    ]),
    ("Perplexity / Benchmarks", [
        "ppl_stride", "ppl_output_type",
        "hellaswag", "hellaswag_tasks",
        "winogrande", "winogrande_tasks",
        "multiple_choice", "multiple_choice_tasks",
        "kl_divergence", "check",
    ]),
    ("Prompt / Input", [
        "prompt", "prompt_file", "path_prompt_cache", "input_prefix", "input_suffix",
        "input_prefix_bos", "antiprompt", "in_files",
    ]),
    ("Retrieval / Passkey", [
        "context_files", "chunk_size", "chunk_separator",
        "n_junk", "i_pos",
    ]),
    ("RoPE / YaRN", [
        "rope_freq_base", "rope_freq_scale", "rope_scaling_type",
        "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_fast", "yarn_beta_slow", "yarn_orig_ctx",
    ]),
    ("Server", [
        "port", "hostname", "public_path", "api_prefix",
        "timeout_read", "timeout_write", "n_threads_http",
        "n_cache_reuse", "cache_prompt", "cache_idle_slots", "n_ctx_checkpoints", "checkpoint_every_nt", "cache_ram_mib",
        "chat_template", "use_jinja", "enable_chat_template",
        "reasoning_format", "enable_reasoning",
        "prefill_assistant", "sleep_idle_seconds",
        "api_keys", "ssl_file_key", "ssl_file_cert",
        "default_template_kwargs",
        "webui", "webui_mcp_proxy", "webui_config_json",
        "endpoint_slots", "endpoint_props", "endpoint_metrics",
        "models_dir", "models_preset", "models_max", "models_autoload",
        "log_json", "slot_save_path", "media_path", "slot_prompt_similarity",
    ]),
    ("Sub-params", [
        "sampling", "speculative", "vocoder", "diffusion",
    ]),
    ("Tensor Buffer Overrides", [
        "tensor_buft_overrides",
    ]),
]


def _clean_default(raw: str) -> str:
    """Clean a C++ default value into a more readable form."""
    if not raw:
        return ""
    raw = raw.strip()
    # Remove C++ float suffix (e.g. 0.5f -> 0.5)
    raw = re.sub(r"(\d+\.\d+)f\b", r"\1", raw)
    # Keep well-known constants as-is
    if raw == "LLAMA_DEFAULT_SEED":
        return "LLAMA_DEFAULT_SEED"
    # Empty brace initializer -> []
    if raw in ("{}", "{ }"):
        return "[]"
    # Empty string literal
    if raw == '""':
        return '""'
    return raw


def _format_type(python_type: str) -> str:
    """Format a Python type annotation for display."""
    if not python_type:
        return ""
    # Strip the module prefix
    python_type = python_type.replace("xllamacpp.", "")
    return python_type


def _build_linkify_pattern(linkable_classes: set[str]) -> re.Pattern | None:
    """Build a compiled regex that matches any linkable class name.

    Returns *None* when there are no linkable classes.
    """
    if not linkable_classes:
        return None
    # Sort by length descending so longer names match first
    sorted_names = sorted(linkable_classes, key=len, reverse=True)
    return re.compile("|".join(re.escape(n) for n in sorted_names))


def _linkify_type(text: str, pattern: re.Pattern | None) -> str:
    """Replace class names in *text* with Markdown anchor links.

    Example::

        'CommonParamsSampling' -> '[CommonParamsSampling](#commonparamssampling)'
        'list[CommonAdapterLoraInfo]' -> 'list[[CommonAdapterLoraInfo](#commonadapterlorainfo)]'
    """
    if not text or pattern is None:
        return text

    def _replace(m: re.Match) -> str:
        cls_name = m.group(0)
        return f"[{cls_name}](#{cls_name.lower()})"

    return pattern.sub(_replace, text)


def generate_markdown(
    pyx_classes: dict[str, ClassInfo],
    cpp_structs: dict[str, dict[str, CppFieldInfo]],
    sort_fields: bool = True,
) -> str:
    """Generate the full Markdown document."""
    out: list[str] = []

    out.append("# xllamacpp Parameters Reference\n")
    out.append(
        "This document is auto-generated from the Cython wrapper (`.pyx`) and the C++ `common.h` header.\n"
        "It lists **all Python-accessible properties** of each configuration class, "
        "along with their types, default values, and descriptions.\n"
    )
    out.append("> **Regenerate**: `python scripts/generate_params_doc.py`\n")

    # Table of contents
    out.append("## Table of Contents\n")
    for cls_name in CLASS_ORDER:
        if cls_name in pyx_classes:
            anchor = cls_name.lower()
            out.append(f"- [{cls_name}](#{anchor})")
    out.append("")

    # Collect all documented class names for cross-reference links
    linkable_classes = {name for name in CLASS_ORDER if name in pyx_classes}
    linkify_pattern = _build_linkify_pattern(linkable_classes)

    for cls_name in CLASS_ORDER:
        cls = pyx_classes.get(cls_name)
        if cls is None:
            continue

        cpp_fields = cpp_structs.get(cls_name, {})
        desc = CLASS_DESCRIPTIONS.get(cls_name, "")

        out.append(f"---\n")
        out.append(f"## {cls_name}\n")
        if desc:
            out.append(f"{desc}\n")

        # Collect all property names for this class
        all_prop_names = {p.name for p in cls.properties}

        # CommonParams uses grouped output
        if cls_name == "CommonParams":
            grouped_names: set[str] = set()
            for group_name, group_fields in COMMON_PARAMS_GROUPS:
                # Filter to properties that actually exist
                existing = [f for f in group_fields if f in all_prop_names]
                if not existing:
                    continue
                grouped_names.update(existing)

                out.append(f"### {group_name}\n")
                out.append("| Property | Type | Default | R/W | Description |")
                out.append("|:---------|:-----|:--------|:---:|:------------|")

                if sort_fields:
                    existing = sorted(existing)

                for prop_name in existing:
                    prop = next(p for p in cls.properties if p.name == prop_name)
                    cpp = cpp_fields.get(prop_name)
                    _write_prop_row(out, prop, cpp, linkify_pattern)

                out.append("")

            # Ungrouped properties
            ungrouped = [p for p in cls.properties if p.name not in grouped_names]
            if ungrouped:
                if sort_fields:
                    ungrouped = sorted(ungrouped, key=lambda p: p.name)
                out.append("### Other\n")
                out.append("| Property | Type | Default | R/W | Description |")
                out.append("|:---------|:-----|:--------|:---:|:------------|")
                for prop in ungrouped:
                    cpp = cpp_fields.get(prop.name)
                    _write_prop_row(out, prop, cpp, linkify_pattern)
                out.append("")
        else:
            # Non-CommonParams classes: flat table
            props = cls.properties
            if sort_fields:
                props = sorted(props, key=lambda p: p.name)
            out.append("| Property | Type | Default | R/W | Description |")
            out.append("|:---------|:-----|:--------|:---:|:------------|")
            for prop in props:
                cpp = cpp_fields.get(prop.name)
                _write_prop_row(out, prop, cpp, linkify_pattern)
            out.append("")

    return "\n".join(out)


def _write_prop_row(
    out: list[str],
    prop: PropertyInfo,
    cpp: CppFieldInfo | None,
    linkify_pattern: re.Pattern | None = None,
):
    """Append a single property row to the Markdown table."""
    ptype = _format_type(prop.python_type)
    default = ""
    desc = prop.docstring

    if cpp:
        default = _clean_default(cpp.default)
        # If the .pyx docstring is empty or shorter, prefer the C++ comment
        if cpp.comment and (not desc or len(desc) < len(cpp.comment)):
            desc = cpp.comment

    rw = "R/W" if prop.has_setter else "R"
    # Escape pipe characters for Markdown tables
    desc = desc.replace("|", "\\|")
    default = default.replace("|", "\\|")

    # Add cross-reference links for documented class names
    ptype = _linkify_type(ptype, linkify_pattern)
    desc = _linkify_type(desc, linkify_pattern)

    out.append(f"| `{prop.name}` | {ptype} | `{default}` | {rw} | {desc} |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown documentation for CommonParams and related classes."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="params_reference.md",
        help="Output file path (default: params_reference.md in cwd). Use '-' for stdout.",
    )
    parser.add_argument(
        "--pyx",
        type=str,
        default=str(PYX_FILE),
        help=f"Path to the .pyx file (default: {PYX_FILE})",
    )
    parser.add_argument(
        "--header",
        type=str,
        default=str(COMMON_H),
        help=f"Path to common.h (default: {COMMON_H})",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        default=False,
        help="Disable alphabetical sorting of fields within each section.",
    )
    args = parser.parse_args()

    pyx_path = Path(args.pyx)
    h_path = Path(args.header)

    if not pyx_path.exists():
        print(f"Error: .pyx file not found: {pyx_path}", file=sys.stderr)
        sys.exit(1)
    if not h_path.exists():
        print(f"Error: common.h not found: {h_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {pyx_path} ...", file=sys.stderr)
    pyx_classes = parse_pyx(pyx_path)
    print(f"  Found {len(pyx_classes)} classes:", file=sys.stderr)
    for name, cls in pyx_classes.items():
        print(f"    {name}: {len(cls.properties)} properties", file=sys.stderr)

    print(f"Parsing {h_path} ...", file=sys.stderr)
    cpp_structs = parse_common_h(h_path)
    print(f"  Found {len(cpp_structs)} structs:", file=sys.stderr)
    for name, fields in cpp_structs.items():
        print(f"    {name}: {len(fields)} fields", file=sys.stderr)

    # Warn about properties that have no description from either source
    for cls_name, cls in pyx_classes.items():
        cpp_fields = cpp_structs.get(cls_name, {})
        for p in cls.properties:
            cpp = cpp_fields.get(p.name)
            if not p.docstring and (not cpp or not cpp.comment):
                print(f"  Warning: no description for {cls_name}.{p.name}", file=sys.stderr)

    md = generate_markdown(pyx_classes, cpp_structs, sort_fields=not args.no_sort)

    if args.output == "-":
        print(md)
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"\nDocumentation written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
