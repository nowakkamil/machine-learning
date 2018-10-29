class InvalidValueOfArgumentException(Exception):
    """
    Attributes:
        message -- cause of the exception
    """

    def __init__(self, message):
        self.message = message

class ArgumentsNotEqualException(Exception):
    """
    Attributes:
        type -- type of arguments
        message -- cause of the exception
    """

    def __init__(self, type, message):
        self.type = type
        self.message = message

class EmptyContainerException(Exception):
    """
    Attributes:
        type -- type of container
        message -- cause of the exception
    """

    def __init__(self, type, message):
        self.type = type
        self.message = message
