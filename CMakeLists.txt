cmake_minimum_required(VERSION 3.10)
project(NBody_Ubuntu_18)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(CUDA REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)

include_directories(src ${SDL2_INCLUDE_DIR})

cuda_add_executable(NBodySimulation
    src/LTexture.cpp
    src/LTexture.h
    src/main.cpp
    src/main.h
    src/NBody.cuh
    src/NBodyCuda.cu)

target_link_libraries(NBodySimulation ${SDL2_LIBRARY} ${SDL2_TTF_LIBRARY})

install(TARGETS NBodySimulation DESTINATION ${CMAKE_BINARY_DIR}/bin)
install(FILES ${CMAKE_SOURCE_DIR}/resource/lazy.ttf DESTINATION ${CMAKE_BINARY_DIR}/bin)


