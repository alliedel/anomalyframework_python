cmake_minimum_required(VERSION 2.8)
project(liblinear)


set(BLAS_SOURCES
        blas/blas.h
        blas/blasp.h
        blas/daxpy.c
        blas/ddot.c
        blas/dnrm2.c
        blas/dscal.c)

set(LIBLINEAR_SOURCES ${BLAS_SOURCES}
        predict.c
        train.c
        linear.cpp
        tron.cpp
        linear.h
        tron.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(multicore-liblinear ${LIBLINEAR_SOURCES})

add_executable(train train.c)
target_link_libraries(train multicore-liblinear)

add_executable(predict predict.c)
target_link_libraries(predict multicore-liblinear)
