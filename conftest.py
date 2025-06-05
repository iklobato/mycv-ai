# pytest-sugar configuration for better output
def pytest_configure(config):
    """Configure pytest-sugar and other display options."""
    # Ensure we have clean output with pytest-sugar
    config.option.verbose = 0  # Turn off verbose mode
    
    # Configure terminal reporting
    if hasattr(config, '_tmp_path_factory'):
        config.option.tb = 'short'

def pytest_collection_modifyitems(config, items):
    """Modify test items for better organization."""
    # Group tests by module for cleaner output
    for item in items:
        # Add module-level markers for better grouping
        module_name = item.module.__name__.split('.')[-1]
        if 'unit' in module_name:
            item.add_marker('unit')
        elif 'integration' in module_name:
            item.add_marker('integration')

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Customize terminal summary for cleaner output."""
    # This helps pytest-sugar provide better summary information
    pass 