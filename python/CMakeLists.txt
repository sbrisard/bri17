find_package(pybind11 REQUIRED)

add_compile_definitions(
        __AUTHOR__="\"\"${METADATA_AUTHOR}\"\""
        __VERSION__="\"\"${METADATA_VERSION}\"\""
)

pybind11_add_module(pybri17 pybri17.cpp)
target_link_libraries(pybri17 PRIVATE bri17)