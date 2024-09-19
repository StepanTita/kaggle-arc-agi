def parse_output(text):
    # Remove any text before and after the actual grid
    grid_match = re.search(r'\[(\s*\[.*?\])+\s*\]', text, re.DOTALL)
    if grid_match:
        text = grid_match.group()
    
    # Replace single quotes with double quotes
    text = text.replace("'", '"')
    
    # Remove any non-JSON characters
    text = re.sub(r'[^\[\],\d\s]', '', text)
    
    try:
        # Try to parse as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If JSON parsing fails, try a more lenient approach
        rows = re.findall(r'\[(.*?)\]', text)
        grid = []
        for row in rows:
            try:
                grid.append([int(x.strip()) for x in row.split(',') if x.strip()])
            except ValueError:
                # If we can't parse a row, return None
                return None
        return grid if grid else None