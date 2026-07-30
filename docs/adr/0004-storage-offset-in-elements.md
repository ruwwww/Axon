# Tensor storage offset stored in elements, not bytes

Tensor carries an `int64_t storage_offset_` field denominated in **elements** of the tensor's dtype, not bytes. This aligns with the existing convention that shape and strides are already element-counted. The alternative (byte offset) would require repeated `sizeof(T)` arithmetic and `char*` casts across the codebase, and is unnecessary because Axon does not support in-place dtype reinterpretation views.

This offset is zero for all tensors created by the allocator; view operations such as `slice`, `narrow`, or `as_strided` set it to a non-zero value to avoid copying storage.
