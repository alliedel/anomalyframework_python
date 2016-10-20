cmake_minimum_required(VERSION 2.8)
project(per_shuffle)

add_executable(score_shuffle
        score_shuffle.cpp
        )
target_link_libraries(score_shuffle multicore-liblinear)
