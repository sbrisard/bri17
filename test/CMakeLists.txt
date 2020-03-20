find_package(PkgConfig REQUIRED)
pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)

add_executable(test_bri17 test_bri17.cpp)
target_link_libraries(test_bri17 PRIVATE bri17 PkgConfig::deps)
add_test(NAME test_bri17 COMMAND test_bri17)
