cmake_minimum_required(VERSION 3.13)

project("example_bri17")

find_package(bri17 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_link_libraries(${PROJECT_NAME} bri17::bri17)
