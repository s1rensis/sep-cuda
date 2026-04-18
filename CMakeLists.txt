cmake_minimum_required(VERSION 3.20)

project(sep-cuda LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(ExternalProject)

option(SEPCUDA_BUILD_TESTS "Build tests" ON)
set(SEPCUDA_REFERENCE_SEP_SOURCE_DIR "" CACHE PATH "Optional path to CPU SEP source tree for comparison tests")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/dist" CACHE PATH "Install prefix" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

add_library(sep_cuda SHARED
  src/background.cpp
  src/cuda_background.cu
  src/util.cpp
)

target_include_directories(sep_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(sep_cuda PRIVATE CUDA::cudart m)

set_target_properties(sep_cuda PROPERTIES
  OUTPUT_NAME _sep_cuda
  EXPORT_NAME SEPCUDA
  C_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS sep_cuda
  EXPORT SEPCUDATargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES src/sep_cuda.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SEPCUDAConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SEPCUDAConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SEPCUDA
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/SEPCUDAConfigVersion.cmake
  VERSION 1.0.0
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/SEPCUDAConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/SEPCUDAConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SEPCUDA
)

install(EXPORT SEPCUDATargets
  FILE SEPCUDATargets.cmake
  NAMESPACE SEPCUDA::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SEPCUDA
)

add_custom_target(dist
  COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --prefix ${CMAKE_INSTALL_PREFIX}
  DEPENDS sep_cuda
)

if(SEPCUDA_BUILD_TESTS)
  enable_testing()

  add_executable(test_smoke tests/test_smoke.cpp)
  target_link_libraries(test_smoke PRIVATE sep_cuda m)
  add_test(NAME sepcuda_smoke COMMAND test_smoke)
  set_tests_properties(sepcuda_smoke PROPERTIES SKIP_RETURN_CODE 77)

  if(SEPCUDA_REFERENCE_SEP_SOURCE_DIR)
    set(SEPCUDA_REFERENCE_SEP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/sep_reference)
    set(SEPCUDA_REFERENCE_SEP_LIBRARY ${SEPCUDA_REFERENCE_SEP_BINARY_DIR}/libsep.so)

    ExternalProject_Add(
      sep_reference_build
      SOURCE_DIR ${SEPCUDA_REFERENCE_SEP_SOURCE_DIR}
      BINARY_DIR ${SEPCUDA_REFERENCE_SEP_BINARY_DIR}
      CONFIGURE_COMMAND ${CMAKE_COMMAND} -S ${SEPCUDA_REFERENCE_SEP_SOURCE_DIR} -B ${SEPCUDA_REFERENCE_SEP_BINARY_DIR} -DCMAKE_POLICY_VERSION_MINIMUM=3.5
      BUILD_COMMAND ${CMAKE_COMMAND} --build ${SEPCUDA_REFERENCE_SEP_BINARY_DIR} -j4
      INSTALL_COMMAND ""
      BUILD_BYPRODUCTS ${SEPCUDA_REFERENCE_SEP_LIBRARY}
    )

    add_executable(test_compare tests/test_compare.cpp)
    target_link_libraries(test_compare PRIVATE dl m)
    target_include_directories(test_compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_dependencies(test_compare sep_reference_build)
    add_test(
      NAME sepcuda_compare_reference
      COMMAND test_compare $<TARGET_FILE:sep_cuda> ${SEPCUDA_REFERENCE_SEP_LIBRARY}
    )
    set_tests_properties(sepcuda_compare_reference PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()
