add_executable(inference inference.cpp)
target_include_directories(inference PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS})
target_link_libraries(inference PRIVATE ${ONNX_RUNTIME_LIB})



add_subdirectory(libClusterDecayer)

install(EXPORT ClusterDecayerTargets
    FILE ClusterDecayerTargets.cmake
    DESTINATION lib/cmake/ClusterDecayer
)


add_executable(test_cluster_decayer test_cluster_decayer.cpp)
target_link_libraries(test_cluster_decayer PRIVATE ClusterDecayer)