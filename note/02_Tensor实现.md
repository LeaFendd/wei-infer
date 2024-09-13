`Tensor`类使用`Buffer`作为数据的容器。
- `allocate()`: 调用`Allocator`分配内存。
- `assign()`: 使用一个已有的`Buffer`进行赋值。
- `init_buffer()`: 根据需要申请Buffer，或者把指定的数据封装到Buffer。
- `ctor()`: 构造Tensor时有两种情况，一种是数据已经在内存中，指针为ptr，只需要使用一个Buffer把数据封装起来；另一种是数据还没有初始化，需要使用`Allocator`分配内存。

- `to()`: 在目标设备端新建一个`Buffer`，复制数据，把`Tensor`的`buffer_`指向新的`Buffer`。