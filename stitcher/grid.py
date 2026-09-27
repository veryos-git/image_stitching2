"""Read logical grid positions from configurable tile filename templates."""
import re

DEFAULT_TEMPLATE = 'prefix_{rNN}_{cNN}.png'
NAME_HELP = ('Classic stitching requires image names including the row and column, '
             'for example tile_r00_c09.png. Adjust the filename template to match your names. '
             'For random images without row and column information, use the separate '
             'Pipeline comparison tool (/compare) with Unordered mode.')


def compile_template(template):
    if not isinstance(template, str) or not template or len(template) > 256:
        raise ValueError('Enter a filename template of 1–256 characters.')
    parts = re.split(r'(prefix|\{rNN\}|\{cNN\}|\{row\}|\{col\})', template)
    tokens = {'{rNN}': ('row', r'r(?P<row>[0-9]+)'),
              '{cNN}': ('col', r'c(?P<col>[0-9]+)'),
              '{row}': ('row', r'(?P<row>[0-9]+)'),
              '{col}': ('col', r'(?P<col>[0-9]+)')}
    found = set()
    pattern = []
    for part in parts:
        if part in tokens:
            name, expression = tokens[part]
            if name in found:
                raise ValueError('The template must contain exactly one row and one column token.')
            found.add(name)
            pattern.append(expression)
        elif part == 'prefix':
            pattern.append(r'.+?')
        else:
            if '{' in part or '}' in part:
                raise ValueError('Use {rNN} and {cNN}, or {row} and {col}, as position tokens.')
            pattern.append(re.escape(part))
    if found != {'row', 'col'}:
        raise ValueError('The template must contain one row and one column token.')
    return re.compile(''.join(pattern), re.IGNORECASE)


def inspect_names(names, template=DEFAULT_TEMPLATE):
    pattern = compile_template(template)
    positions, errors, seen = [], [], {}
    for name in names:
        match = pattern.fullmatch(name)
        if match is None:
            positions.append(None)
            errors.append(f'{name}: row and column could not be detected.')
            continue
        position = (int(match['row']), int(match['col']))
        positions.append(position)
        if position in seen:
            errors.append(f'{name} and {seen[position]} have the same row {position[0]}, column {position[1]}.')
        seen[position] = name
    return dict(positions=positions, errors=errors, valid=not errors)


def require_positions(names, template=DEFAULT_TEMPLATE):
    result = inspect_names(names, template)
    if result['errors']:
        raise ValueError('\n'.join(result['errors'][:8]) + '\n\n' + NAME_HELP)
    return result['positions']
