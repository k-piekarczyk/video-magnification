__all__ = ["closest_power_of_2", "is_power_of_2"]


def closest_power_of_2(x: int):
    """
    Returns a closest power of 2 that is equal or bigger than `x`

    Credits to this user on SO: https://stackoverflow.com/a/14267825

    Works by some clever bit shifting magic.
    Since in a bit representation of an intiger, if only one bit is set to 1 and the rest is 0,
    then the resulting integer is by the very nature of a base 2 number system is a power of 2.
    `x-1` ensures, that any power of 2 input is reduced by one, making it's bit representation
    shorter, so shifting a bit of `0b1` to the left by the length of the bit representation of `x-1`
    returns the first power of 2 bigger than `x-1`. If `x` is already a power of 2, then that
    bitshift results in `x`.
    """
    return 1 << (x - 1).bit_length()


def is_power_of_2(x: int):
    """
    Returns true if `x` is a power of 2.

    This is in essence a single step from the Brian Kernighan's algorithm for
    counting set bits in a number. Since integer powers of 2 only have 1 set bit,
    and the algorithm only loops the same amount of times as the number of set bits.
    """
    return x & x - 1 == 0
