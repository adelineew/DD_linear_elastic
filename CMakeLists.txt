##
#  CMake script for the linear elastic principle program:
##


# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
FIND_PACKAGE(deal.II 8.5.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()


 INCLUDE_DIRECTORIES(
  include
  ${CMAKE_SOURCE_DIR}/include
  )

ADD_LIBRARY(MMASolver ${CMAKE_SOURCE_DIR}/source/MMASolver.cc)
DEAL_II_SETUP_TARGET(MMASolver)

ADD_LIBRARY(ConstituitiveDD ${CMAKE_SOURCE_DIR}/source/ConstituitiveDD.cc)
DEAL_II_SETUP_TARGET(ConstituitiveDD)

ADD_LIBRARY(LinearElasticity  ${CMAKE_SOURCE_DIR}/source/LinearElasticity_implicit.cc)
DEAL_II_SETUP_TARGET(LinearElasticity)

ADD_LIBRARY(Constituitive ${CMAKE_SOURCE_DIR}/source/Constituitive.cc)
DEAL_II_SETUP_TARGET(Constituitive)

TARGET_LINK_LIBRARIES(LinearElasticity Constituitive)
TARGET_LINK_LIBRARIES(LinearElasticity MMASolver)
TARGET_LINK_LIBRARIES(LinearElasticity ConstituitiveDD)

ADD_EXECUTABLE(run_stuff ${CMAKE_SOURCE_DIR}/source/run_stuff.cc)
DEAL_II_SETUP_TARGET(run_stuff)
TARGET_LINK_LIBRARIES(run_stuff LinearElasticity)

#ADD_EXECUTABLE(restart_run ${CMAKE_SOURCE_DIR}/source/restart_run.cc)
#DEAL_II_SETUP_TARGET(restart_run)
#TARGET_LINK_LIBRARIES(restart_run LEimplicit)

# Define a distclean target to remove every generated file:
ADD_CUSTOM_TARGET(distclean
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
  COMMAND ${CMAKE_COMMAND} -E remove_directory CMakeFiles
  COMMAND ${CMAKE_COMMAND} -E remove CMakeCache.txt cmake_install.cmake Makefile
  COMMENT "distclean invoked"
  )


