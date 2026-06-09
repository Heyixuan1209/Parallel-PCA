cmake_minimum_required(VERSION 3.13)
project(PCA C)

set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Place runtime binaries into bin/ at source root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")

find_package(MPI REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS)
find_package(OpenMP)
find_library(LAPACKE_LIB NAMES lapacke)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CORE_SRCS
    src/core/matrix.c
    src/core/jacobi_ser.c
    src/core/jacobi_par.c
    src/core/pca.c
    src/core/utils.c
)

add_executable(pca_eig_ser src/benchmark/eig_ser.c ${CORE_SRCS})
add_executable(pca_eig_par_pro src/benchmark/eig_par_pro.c ${CORE_SRCS})
add_executable(pca_eig_par_thr src/benchmark/eig_par_thr.c ${CORE_SRCS})
add_executable(pca_svd_ser src/benchmark/svd_ser.c ${CORE_SRCS})
add_executable(pca_svd_par src/benchmark/svd_par.c ${CORE_SRCS})

if(TARGET LAPACK::LAPACK)
  set(LAPACK_LIBS LAPACK::LAPACK)
else()
  set(LAPACK_LIBS ${LAPACK_LIBRARIES})
endif()

if(TARGET BLAS::BLAS)
  set(BLAS_LIBS BLAS::BLAS)
else()
  set(BLAS_LIBS ${BLAS_LIBRARIES})
endif()

foreach(tgt pca_eig_ser pca_eig_par_pro pca_eig_par_thr pca_svd_ser pca_svd_par)
  target_link_libraries(${tgt} PUBLIC MPI::MPI_C ${LAPACK_LIBS} ${BLAS_LIBS} ${LAPACKE_LIB} m)
endforeach()

if(OpenMP_C_FOUND)
  target_link_libraries(pca_eig_par_thr PUBLIC OpenMP::OpenMP_C)
else()
  message(WARNING "OpenMP for C was not found. pca_eig_par_thr will compile, but root_omp_jacobi will effectively run serially.")
endif()
