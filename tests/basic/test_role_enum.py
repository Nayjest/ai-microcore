import sys

import pytest

from microcore.message_types import Role


def test_role():
    assert Role.USER == "user"
    assert Role.USER != Role.SYSTEM
    assert isinstance(Role.USER, Role)
    assert isinstance(Role.USER, str)
    assert (Role.USER + 'str') == "userstr"
    assert ','.join(Role) == "system,user,assistant"


def test___str__():
    assert str(Role.SYSTEM) == "system"  # __str__
    assert str(Role.ASSISTANT) == "assistant"
    assert f"{Role.USER}" == "user"
    assert "%s" % Role.USER == "user"


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="""
    in Python 3.11.X and older __contains__ will raise TypeError:
    unsupported operand type(s) for 'in': 'str' and 'EnumMeta'
    """
)
def test_in():
    assert Role.USER in Role
    assert "user" in Role
