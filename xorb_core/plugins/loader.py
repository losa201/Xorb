import importlib.metadata

def load_plugins(group_name: str):
    """Load plugins for a given group."""
    plugins = []
    for entry_point in importlib.metadata.entry_points(group=group_name):
        try:
            plugin = entry_point.load()
            plugins.append(plugin)
        except Exception as e:
            print(f"Failed to load plugin {entry_point.name}: {e}")
    return plugins
