class MetricMeter:
    """Tracks a streaming metric using observed values.

    This class is useful for monitoring metrics during training loops.

    Attributes:
        count: The number of values that have been observed.
        last_val: The most recent value that was passed to `update`.
        sum: The cumulative sum of all values observed so far.
        avg: The running average of all observed values.
    """

    def __init__(self) -> None:
        """Initializes the MetricMeter.
        """
        self.count = 0

        self.last_val = 0.0
        self.avg = 0.0
        self.sum = 0.0

    def update(self, val, n: int = 1) -> None:
        """Updates the meter with new value or values.

        This function handles batches of data by using the variable 'n'.

        Args:
            val: The latest value to include in the metric statistics.
            n: Number of samples represented by 'val'.
        """
        self.count += n

        self.last_val = val
        self.sum += val * n
        self.avg = self.sum / self.count

    def reset(self) -> None:
        """Resets the MetricMeter.
        """
        self.count = 0

        self.last_val = 0.0
        self.avg = 0.0
        self.sum = 0.0
