project("bri17_example")

#find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(bri17 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} src/bri17_example.cc)
target_link_libraries(${PROJECT_NAME} bri17::bri17)
