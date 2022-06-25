from .powers import closest_power_of_2, is_power_of_2


__all__ = ["get_max_pyramid_depth"]


def get_max_pyramid_depth(height: int, width: int, min_unit_size: int = 8) -> int:
    """
    Returns the maximum depth of a pyramid for a given height and width (adjusts them to the closes power of 2)
    """
    if min_unit_size <= 2 or not is_power_of_2(min_unit_size):
        raise Exception(f"'min_unit_size' has to be at least 2, and be divisible by 2. Provided value: {min_unit_size}")

    adjusted_height = closest_power_of_2(height)
    adjusted_width = closest_power_of_2(width)

    smaller_adjusted_side_length = min(adjusted_height, adjusted_width)
    if smaller_adjusted_side_length <= min_unit_size:
        raise Exception(
            f"Impossible to build a Laplacian pyramid with the given 'min_unit_size' of {min_unit_size} from the given 'frame'"
        )

    max_depth = smaller_adjusted_side_length.bit_length() - min_unit_size.bit_length()

    return max_depth
