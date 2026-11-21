add_library(marin_dlt_truncation INTERFACE)

target_include_directories(marin_dlt_truncation INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include/marin_dlt_truncation>)
target_compile_features(marin_dlt_truncation INTERFACE cxx_std_17)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION include/marin_dlt_truncation FILES_MATCHING PATTERN "*.hpp")
