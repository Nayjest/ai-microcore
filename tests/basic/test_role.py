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
