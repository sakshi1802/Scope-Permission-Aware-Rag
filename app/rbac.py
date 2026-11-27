ROLES = {
    "intern": "Intern",
    "engineer": "Software Engineer",
    "eng_manager": "Engineering Manager",
    "finance": "Finance",
    "hr": "People Ops",
}


def resolve_allowed_roles(declared: list[str]) -> list[str]:
    if not declared:
        raise ValueError("allowed_roles must not be empty")
    if "all" in declared:
        return list(ROLES)
    unknown = [r for r in declared if r not in ROLES]
    if unknown:
        raise ValueError(f"unknown roles in allowed_roles: {unknown}")
    return sorted(set(declared))


def validate_role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown role: {role!r}")
    return role