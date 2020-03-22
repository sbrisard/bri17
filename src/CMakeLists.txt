find_library(MATH_LIBRARY m)

#add_library(bri17 SHARED
#  bri17.cpp
#  ../include/bri17/bri17.h)

add_executable(bri17 bri17.cpp ../include/bri17/bri17.h)

target_include_directories(bri17 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}../include>
  $<INSTALL_INTERFACE:include>)

target_include_directories(bri17 PUBLIC ../include)
if (MATH_LIBRARY)
  target_link_libraries(bri17 ${MATH_LIBRARY})
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (bri17 Eigen3::Eigen)

# install(TARGETS bri17 LIBRARY DESTINATION lib)
# install(DIRECTORY ../include/bri17 DESTINATION include)
