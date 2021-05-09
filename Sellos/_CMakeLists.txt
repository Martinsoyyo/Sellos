cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(Sellos)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(Sellos main.cpp Parser.cpp )

# Precompiled header pch.h
target_precompile_headers(Sellos PUBLIC pch.h)

# Muestra el tiempo de compilación de cada archivo
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")

target_link_libraries(Sellos "${TORCH_LIBRARIES}")
set_property(TARGET Sellos PROPERTY CXX_STANDARD 14)


if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET Sellos
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:torch-test>)
endif (MSVC)