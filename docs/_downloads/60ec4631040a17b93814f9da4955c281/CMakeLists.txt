cmake_minimum_required(VERSION 3.13)

project("bri17_example")

find_package(bri17 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} src/bri17_example.cc)
target_link_libraries(${PROJECT_NAME} bri17::bri17)
