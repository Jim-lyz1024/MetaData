class Registry:
    """A registry to map strings to classes or functions.
    
    Args:
        name (str): registry name
    """
    
    def __init__(self, name):
        self._name = name
        self._obj_map = {}
        
    def _do_register(self, name, obj):
        assert name not in self._obj_map, (
            "An object named '{}' was already registered "
            "in '{}' registry!".format(name, self._name)
        )
        self._obj_map[name] = obj
        
    def register(self, obj=None):
        """Register an object.
        
        Args:
            obj (object, optional): object to register
        """
        if obj is None:
            # Used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco
            
        # Used as a function call
        name = obj.__name__
        self._do_register(name, obj)
        
    def get(self, name):
        """Get registered object.
        
        Args:
            name (str): registered name
            
        Returns:
            object: registered object
        """
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name
                )
            )
        return ret
        
    def registered_names(self):
        """Get registered names."""
        return list(self._obj_map.keys())

def check_availability(requested, available):
    """Check if an object is available in a list.
    
    Args:
        requested (str): requested object name
        available (list): list of available objects
    """
    if requested not in available:
        raise ValueError(
            "'{}' is not available. Available options are: {}".format(
                requested, available
            )
        )