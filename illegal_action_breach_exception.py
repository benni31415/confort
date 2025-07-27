class IllegalActionBreachException(Exception):
    
    def __init__(self, occurrences, sample_size, limit):
        self.message = f"Detected too many illegal actions in past records " +\
              "(detected: {occurrences}, sample size: {sample_size}, allowed: {limit})"