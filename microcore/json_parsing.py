import json
import re
from enum import Enum

from .types import BadAIJsonAnswer


def safe_remove_outer_json_wrapper(input_string: str) -> str:
    if input_string.endswith("```"):
        if input_string.startswith("```json"):
            return input_string[7:-3].strip()
        if input_string.startswith("```"):
            return input_string[3:-3].strip()
    return input_string


def simple_json_format_check(input_string: str):
    for opener, close in [("{", "}"), ("[", "]"), ('"', '"')]:
        if input_string.startswith(opener) and input_string.endswith(close):
            return True
    if input_string.isdigit() or input_string in ["true", "false", "null"]:
        return True
    try:
        float(input_string)
        return True
    except ValueError:
        ...
    return False


class ExtractStrategy(str, Enum):
    FIRST = "first"
    LAST = "last"
    OUTER = "outer"


def extract_block(
    input_string: str,
    block_begin: str,
    block_end: str,
    include_wrapper: bool = False,
    strategy: ExtractStrategy = ExtractStrategy.OUTER,
) -> str | None:
    """
    Extracts a block of text between two markers from a string.
    """
    assert block_begin and block_end, "Block markers cannot be empty."
    start = -1
    begin_len = 1
    if strategy in (ExtractStrategy.OUTER, ExtractStrategy.FIRST):
        start = input_string.find(block_begin)
        if start == -1:
            return None
        begin_len = len(block_begin)
    if strategy in (ExtractStrategy.OUTER, ExtractStrategy.LAST):
        end = input_string.rfind(block_end, start + begin_len)
    else:  # first
        end = input_string.find(block_end, start + begin_len)
    if end == -1:
        return None
    if strategy == ExtractStrategy.LAST:
        start = input_string.rfind(block_begin, None, end)
        if start == -1:
            return None

    start_capture = start if include_wrapper else start + len(block_begin)
    end_capture = end + len(block_end) if include_wrapper else end

    return input_string[start_capture:end_capture]


def unwrap_json_substring(
    input_string: str, allow_in_text: bool = True, return_original_on_fail: bool = True
) -> str:
    input_string = safe_remove_outer_json_wrapper(str(input_string).strip())
    if simple_json_format_check(input_string):
        return input_string

    if not allow_in_text:
        return input_string if return_original_on_fail else ""

    if (val := extract_block(input_string, "```json", "```", False)) is not None:
        return val.strip()

    # find outermost {} or []
    brace = None
    start = 0
    end = 0
    try:
        start, end = input_string.index("{"), input_string.rindex("}")
        if end > start:
            brace = "{"
    except ValueError:
        ...
    try:
        s_start, s_end = input_string.index("["), input_string.rindex("]")
        if not brace or (s_start < start and s_end > end):
            start = s_start
            end = s_end
            brace = "["
    except ValueError:
        ...

    return (
        input_string[start: end + 1]
        if brace
        else input_string if return_original_on_fail else ""
    )


# pylint: disable=too-many-return-statements
def fix_json(s: str) -> str:
    """
    Fix internal JSON content.
    Note: this function should not be used for valid JSON strings.
    Args:
        s (str): AI-generated JSON string containing errors.
    """

    def between_lines(pattern):
        return (
            rf"({json_obj_before}{pattern}{json_obj_after})"
            rf"|({json_list_before}{pattern}{json_list_after})"
        )

    json_obj_before = (
        r"((?<=\{)|(?<=[\"\d]\,)|"
        r"(?<=null\,)|(?<=true\,)|(?<=false\,)|(?<=[\"\d])|"
        r"(?<=null)|(?<=true)|(?<=false))\s*"
    )
    json_obj_after = r'\s*((?=\})|(?=\"[^"\n]+\"\s*\:\s*))'
    json_list_before = (
        r"((?<=\[)|(?<=[\"\d]\,)|(?<=null\,)|(?<=true\,)|(?<=false\,)|"
        r"(?<=[\"\d])|(?<=null)|(?<=true)|(?<=false))\s*"
    )
    json_list_after = r"\s*((?=[\]\"\d])|(?=true)|(?=false)|(?=null))"
    json_before = rf"({json_obj_before}|{json_list_before})"
    json_after = rf"({json_obj_after}|{json_list_after})"

    try:
        comment = r"(//|\#)[^\n]*\n"
        comments = rf"({comment})+"
        s = re.sub(between_lines(comments), "\n", s)
        return json.dumps(json.loads(s), indent=4)
    except json.JSONDecodeError:
        ...

    try:
        # ... typically added by LLMs to identify that sequence may be continued
        s = re.sub(between_lines(r"\.\.\.\n"), "\n", s)
        return json.dumps(json.loads(s), indent=4)
    except json.JSONDecodeError:
        ...

    try:
        # missing comma between strings on separate lines
        s = re.sub(r"\"\s*\n\s*\"", '",\n"', s)
        return json.dumps(json.loads(s), indent=4)
    except json.JSONDecodeError:
        ...

    try:
        # Redundant trailing comma
        s = re.sub(
            r"((?<=[\"\d])|(?<=null)|(?<=true)|(?<=false))\s*\,(?=\s*[\}\]])", "", s
        )
        return json.dumps(json.loads(s), indent=4)
    except json.JSONDecodeError:
        ...

    try:
        # Fix incorrect quotes
        s = re.sub(rf"({json_before}\')|(\'{json_after})", '"', s)
        s = re.sub(rf"\'\s*\,\s*{json_after}", '", ', s)
        s = re.sub(r"\'\s*\:\s*(?=[\'\"])", '": ', s)
        s = re.sub(r"(?<=[\'\"])\s*\:\s*\'", ': "', s)
        return json.dumps(json.loads(s), indent=4)
    except json.JSONDecodeError:
        ...

    try:
        # Python-style values instead of JSON (inside fields)
        mapping = {"False": "false", "True": "true", "None": "null"}
        for pythonic, jsonic in mapping.items():
            s = re.sub(rf"\"\:\s*{pythonic}(?=\s*[\,\}}])", f'": {jsonic}', s)
        return json.dumps(json.loads(s), indent=4)
    except json.JSONDecodeError:
        ...

    try:
        # Drop inline comments
        s = re.sub(r"\/\*[^\n]*\*\/", "", s)
        return json.dumps(json.loads(s), indent=4)
    except json.JSONDecodeError:
        ...

    # incomplete JSON
    if s.startswith("{") and not s.endswith("}"):
        # count number of "
        if s.count('"') % 2 == 1:
            s += '"'
        s += "}"
    return s


def parse_json(
    input_string: str, raise_errors: bool = True, required_fields: list[str] = None
) -> list | dict | float | int | str:
    """
    Extract and parse JSON from AI-generated string.
    Args:
        input_string (str): String containing JSON.
        raise_errors (bool, optional):
            If True, raises exception on error, otherwise returns False on error.
        required_fields (list, optional): List of expected field names to validate the JSON object.
    Returns:
        list | dict | float | int | str: Parsed JSON data.
    """
    assert isinstance(required_fields, list) or required_fields is None
    try:
        if not input_string:  # empty string
            raise BadAIJsonAnswer(
                "Input string is empty. Cannot parse JSON from an empty string."
            )
        s = unwrap_json_substring(input_string)
        try:
            res = json.loads(s)
        except json.JSONDecodeError:
            res = json.loads(fix_json(s))
        if required_fields:
            if not isinstance(res, dict):
                raise BadAIJsonAnswer(
                    f"Expected a JSON object, but received: {type(res).__name__}"
                )
            for field in required_fields:
                if field not in res:
                    raise BadAIJsonAnswer(
                        f'Required field "{field}" is missing in the JSON object.'
                    )
        return res
    except (json.decoder.JSONDecodeError, BadAIJsonAnswer) as e:
        if raise_errors:
            raise BadAIJsonAnswer(str(e)) from e
        return False
