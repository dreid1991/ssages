project(SSAGES)
cmake_minimum_required(VERSION 2.8.5)

# Default type is release
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

# Resolve dependencies
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Require Boost MPI.
find_package(Boost
	1.55
	REQUIRED
	COMPONENTS mpi serialization
)

# Package src directories
set(LAMMPS_SRC "" CACHE STRING "LAMMPS source directory")
set(DANMD_SRC "" CACHE STRING "DanMD source directory")

include(ExternalProject)

if(LAMMPS_SRC)
	# Add lammps as external project
	ExternalProject_Add(
		lammps
		SOURCE_DIR ${LAMMPS_SRC}
		CONFIGURE_COMMAND ""
		BUILD_COMMAND $(MAKE) mpi mode=shlib 
		CCFLAGS=-I${PROJECT_SOURCE_DIR}/src  
		CCFLAGS+=-I${PROJECT_SOURCE_DIR}/include  
		CCFLAGS+=-I${Boost_INCLUDE_DIRS}     
		CCFLAGS+=-std=c++11                  
		CCFLAGS+=-L${Boost_LIBRARY_DIRS}      
		LIB+=-lboost_mpi                      
		LIB+=-lboost_serialization           
		LIB+=${CMAKE_BINARY_DIR}/libssages.a
		INSTALL_COMMAND ""
		BUILD_IN_SOURCE true
		LOG_BUILD ON
	)

	ExternalProject_Add_Step(
		lammps forceconfigure
		COMMENT "Configuring LAMMPS"
		DEPENDERS build
		DEPENDEES update
		ALWAYS 1
	)

	add_custom_command(
	    TARGET lammps PRE_BUILD
	    COMMAND ln -sf ${PROJECT_SOURCE_DIR}/hooks/lammps/fix_ssages.h ${LAMMPS_SRC}/fix_ssages.h
	    COMMAND touch ${LAMMPS_SRC}/modify.cpp
	)


	# Include directory for the pacakge we're interested in. 
	include_directories("${LAMMPS_SRC}")


	# Add source for appropriate package hook.
	set(HOOKSRC
		hooks/lammps/fix_ssages.cpp
	)
elseif(DANMD_SRC)
    set(ENABLE_DANMD true)
    message(STATUS "md_engine dir is " ${DANMD_SRC})
	ExternalProject_Add(
		danmd
		SOURCE_DIR ${DANMD_SRC}
		PREFIX ${CMAKE_CURRENT_BINARY_DIR}/danmd
		CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
		BUILD_IN_SOURCE true
		INSTALL_DIR ${CMAKE_BINARY_DIR}/danmd
		INSTALL_COMMAND cp ${DANMD_SRC}/src/libSim.so ${CMAKE_CURRENT_BINARY_DIR}/danmd/
		LOG_BUILD ON)
	
	ExternalProject_Add_Step(
		danmd forceconfigure
		COMMENT "Configuring DanMD"
		DEPENDERS build
		DEPENDEES update
		ALWAYS 1
	)

endif()

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}")

# General source files
set(SOURCE
	src/Methods/Meta.cpp
	src/Methods/Umbrella.cpp
	src/Methods/ElasticBand.cpp
	src/Methods/FiniteTempString.cpp
	src/Methods/ForwardFlux.cpp
	src/Validator/RequirementLoader.cpp
	src/JSON/jsoncpp.cpp
	src/JSON/schema.cpp
	src/CVs/CollectiveVariable.cpp
	src/Methods/Method.cpp
)

include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(${Boost_INCLUDE_DIRS})
add_definitions(${MPI_CXX_COMPILE_FLAGS})
link_directories(${MPI_CXX_LIBRARIES})

# SSAGES library.
add_library(libssages ${HOOKSRC} ${SOURCE})
set_target_properties(libssages PROPERTIES PREFIX "")
target_link_libraries(libssages ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(libssages PROPERTIES COMPILE_FLAGS "-Wall -Wextra -fPIC")

# Require C++11.
set_property(TARGET libssages PROPERTY CXX_STANDARD 11)
set_property(TARGET libssages PROPERTY CXX_STANDARD_REQUIRED ON)

# SSAGES executable.
add_executable(ssages
	src/main.cpp
)

# Configure header.
configure_file (
  "${PROJECT_SOURCE_DIR}/src/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
)

# Dependencies. 

if(LAMMPS_SRC)
	add_dependencies(lammps libssages)
	target_link_libraries(ssages libssages ${LAMMPS_SRC}/liblammps_mpi.so)
	add_dependencies(ssages lammps)
elseif(DANMD_SRC)
	add_dependencies(danmd libssages)
	target_link_libraries(ssages libssages ${CMAKE_CURRENT_BINARY_DIR}/danmd/libSim.so)
endif()
set_target_properties(ssages PROPERTIES COMPILE_FLAGS "-Wall -Wextra")

# Configure flags 
# - Globals
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -pthread -std=c++11")

# Add doc subdirectory for building the documentation
add_subdirectory(doc)
