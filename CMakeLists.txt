cmake_minimum_required(VERSION 3.5)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(pluginout)

spiffs_create_partition_image(spiffs spiffs FLASH_IN_PROJECT "${CMAKE_SOURCE_DIR}/spiffs")

