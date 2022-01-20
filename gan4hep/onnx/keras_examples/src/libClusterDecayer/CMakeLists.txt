include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

file(GLOB SrcFiles *.cxx)
add_library(ClusterDecayer SHARED ${SrcFiles})
target_include_directories(ClusterDecayer 
        PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
target_link_libraries(ClusterDecayer PUBLIC ${ONNX_RUNTIME_LIB})

install(TARGETS ClusterDecayer 
        DESTINATION lib
        EXPORT ClusterDecayerTargets)

file(GLOB HeaderFiles include/*.hpp)
install(FILES ${HeaderFiles}
    DESTINATION include/ClusterDecayer)