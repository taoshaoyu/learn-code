- From: https://gflags.github.io/gflags/#using
- CMakeLists.txt

```
find_package(gflags REQUIRED)

# Add an executable
add_executable(hello main.cpp)
target_link_libraries(hello gflags)
```