# set project's name
PROJECT( VanishingPoint )

###############################################################################
# CMake settings
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

# set c++ specific flags
set(CMAKE_CXX_STANDARD 11)

# OpenCV
#FIND_PACKAGE(OpenCV REQUIRED)
include(${PROJECT_SOURCE_DIR}/build/conanbuildinfo.cmake)
conan_basic_setup()

FILE(GLOB_RECURSE HDRS_FILES "src/*.h" "src/*.hpp")
FILE(GLOB_RECURSE SRCS_FILES "src/*.c" "src/*.cpp")

ADD_EXECUTABLE(${PROJECT_NAME} ${SRCS_FILES} ${HDRS_FILES})
#TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CONAN_LIBS})

# learning cv
ADD_EXECUTABLE(cvprac learncv/practice.cpp)
ADD_EXECUTABLE(filestore learncv/filestorage.cpp)
TARGET_LINK_LIBRARIES(cvprac ${CONAN_LIBS})
TARGET_LINK_LIBRARIES(filestore ${CONAN_LIBS})