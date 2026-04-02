from .basic_buffer import BasicBuffer
from .new_buffer import BasicBuffer as NewBasicBuffer

REGISTRY = {
    "basic_buffer": NewBasicBuffer,
}