import json
import re

from .types import BadAIJsonAnswer


def unwrap_json_substring(
        input_string: str, allow_in_text: bool = True, return_original_on_fail: bool = True
) -> str:
    input_string = str(input_string).strip()
    if input_string.endswith("```"):
        if input_string.startswith("```json"):
            return input_string[7:-3].strip()
        if input_string.startswith("```"):
            return input_string[3:-3].strip()

    for opener, close in [("{", "}"), ("[", "]"), ('"', '"')]:
        if input_string.startswith(opener) and input_string.endswith(close):
            return input_string
    if input_string.isdigit() or input_string in ["true", "false", "null"]:
        return input_string

    if not allow_in_text:
        return input_string if return_original_on_fail else ""

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


def fix_json(s: str) -> str:
    """
    Fix internal JSON content
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

    # incomplete JSON
    if s.startswith('{') and not s.endswith('}'):
        # count number of "
        if s.count('"') % 2 == 1:
            s += '"'
        s += '}'
    return s


def parse_json(
        input_string: str, raise_errors: bool = True, required_fields: list[str] = None
) -> list | dict | float | int | str:
    assert isinstance(required_fields, list) or required_fields is None
    try:
        s = unwrap_json_substring(input_string)
        try:
            res = json.loads(s)
        except json.JSONDecodeError:
            res = json.loads(fix_json(s))
        if required_fields:
            if not isinstance(res, dict):
                raise BadAIJsonAnswer("Not an object")
            for field in required_fields:
                if field not in res:
                    raise BadAIJsonAnswer(f'Missing field "{field}"')
        return res
    except json.decoder.JSONDecodeError as e:
        if raise_errors:
            raise BadAIJsonAnswer() from e
        return False
