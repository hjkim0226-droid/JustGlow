# ============================================================================
# JustGlow - After Effects GPU Glow Plugin
# CMake Build Configuration
# ============================================================================

cmake_minimum_required(VERSION 3.20)
project(JustGlow VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Configuration Options
# ============================================================================

# After Effects SDK path
set(AE_SDK_PATH "${CMAKE_SOURCE_DIR}/include/AE_SDK" CACHE PATH "Path to After Effects SDK")

# Check if SDK exists
if(NOT EXISTS "${AE_SDK_PATH}/AE_Effect.h")
    message(WARNING "
    =========================================================================
    After Effects SDK not found at: ${AE_SDK_PATH}

    The SDK should be linked/copied to: ${CMAKE_SOURCE_DIR}/include/AE_SDK

    Expected structure:
    ${AE_SDK_PATH}/
        ├── AE_Effect.h
        ├── AE_EffectCB.h
        ├── AE_EffectGPUSuites.h
        └── ...
    =========================================================================
    ")
endif()

# ============================================================================
# Source Files
# ============================================================================

set(PLUGIN_SOURCES
    src/JustGlow.cpp
    src/JustGlowGPURenderer.cpp
)

set(PLUGIN_HEADERS
    src/JustGlow.h
    src/JustGlowParams.h
    src/JustGlowGPURenderer.h
)

set(SHADER_FILES
    shaders/Common.hlsli
    shaders/Prefilter.hlsl
    shaders/Downsample.hlsl
    shaders/Upsample.hlsl
    shaders/PostProcess.hlsl
    shaders/Composite.hlsl
)

# ============================================================================
# Plugin Target
# ============================================================================

add_library(${PROJECT_NAME} SHARED
    ${PLUGIN_SOURCES}
    ${PLUGIN_HEADERS}
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${AE_SDK_PATH}
    ${AE_SDK_PATH}/SP
    ${AE_SDK_PATH}/Util
)

# ============================================================================
# Windows Configuration
# ============================================================================

if(WIN32)
    # Preprocessor definitions
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        MSWindows
        WIN32
        _WINDOWS
        UNICODE
        _UNICODE
        HAS_DIRECTX=1
    )

    # Add resource and module definition files
    target_sources(${PROJECT_NAME} PRIVATE
        resources/Win/JustGlow.rc
        resources/Win/JustGlow.def
    )

    # Set output extension to .aex
    set_target_properties(${PROJECT_NAME} PROPERTIES
        SUFFIX ".aex"
        PREFIX ""
        OUTPUT_NAME "JustGlow"
    )

    # Link libraries
    target_link_libraries(${PROJECT_NAME} PRIVATE
        d3d12.lib
        dxgi.lib
        d3dcompiler.lib
    )

    # Compiler flags
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W3             # Warning level 3
        /WX-            # Warnings not as errors (for development)
        /MP             # Multi-processor compilation
    )

    # ========================================================================
    # PiPL Resource Compilation
    # ========================================================================

    # Find cl.exe for preprocessing
    find_program(CL_EXECUTABLE cl)

    # PiPL tool path (should be in SDK)
    set(PIPL_TOOL "${AE_SDK_PATH}/../Resources/PiPLTool.exe" CACHE FILEPATH "Path to PiPLTool")

    if(EXISTS "${PIPL_TOOL}")
        set(PIPL_SOURCE "${CMAKE_SOURCE_DIR}/resources/JustGlowPiPL.r")
        set(PIPL_TEMP_RC "${CMAKE_BINARY_DIR}/JustGlowPiPL_temp.rc")

        add_custom_command(
            OUTPUT "${PIPL_TEMP_RC}"
            COMMAND ${CL_EXECUTABLE} /I"${AE_SDK_PATH}" /EP "${PIPL_SOURCE}" > "${CMAKE_BINARY_DIR}/JustGlowPiPL.rr"
            COMMAND "${PIPL_TOOL}" "${CMAKE_BINARY_DIR}/JustGlowPiPL.rr" "${CMAKE_BINARY_DIR}/JustGlowPiPL.rrc"
            COMMAND ${CL_EXECUTABLE} /D "MSWindows" /EP "${CMAKE_BINARY_DIR}/JustGlowPiPL.rrc" > "${PIPL_TEMP_RC}"
            DEPENDS "${PIPL_SOURCE}"
            COMMENT "Compiling PiPL resource"
            VERBATIM
        )

        add_custom_target(CompilePiPL DEPENDS "${PIPL_TEMP_RC}")
        add_dependencies(${PROJECT_NAME} CompilePiPL)
    else()
        message(WARNING "PiPLTool not found. PiPL resource will not be compiled automatically.")
    endif()

    # ========================================================================
    # Shader Compilation
    # ========================================================================

    # Find DirectX Shader Compiler
    find_program(DXC_EXECUTABLE dxc
        HINTS
        "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64"
        "C:/Program Files (x86)/Windows Kits/10/bin/10.0.19041.0/x64"
        "$ENV{VULKAN_SDK}/Bin"
    )

    set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/DirectX_Assets")
    file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

    if(DXC_EXECUTABLE)
        # Compile each shader
        set(SHADER_ENTRY_POINTS
            "Prefilter:main"
            "Downsample:main"
            "Upsample:main"
            "PostProcess:main_anamorphic"
            "Composite:main"
        )

        foreach(SHADER_ENTRY ${SHADER_ENTRY_POINTS})
            string(REPLACE ":" ";" SHADER_PARTS ${SHADER_ENTRY})
            list(GET SHADER_PARTS 0 SHADER_NAME)
            list(GET SHADER_PARTS 1 ENTRY_POINT)

            set(SHADER_INPUT "${CMAKE_SOURCE_DIR}/shaders/${SHADER_NAME}.hlsl")
            set(SHADER_OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.cso")

            add_custom_command(
                OUTPUT "${SHADER_OUTPUT}"
                COMMAND ${DXC_EXECUTABLE}
                    -T cs_6_0
                    -E ${ENTRY_POINT}
                    -Fo "${SHADER_OUTPUT}"
                    -I "${CMAKE_SOURCE_DIR}/shaders"
                    "${SHADER_INPUT}"
                DEPENDS "${SHADER_INPUT}" "${CMAKE_SOURCE_DIR}/shaders/Common.hlsli"
                COMMENT "Compiling shader: ${SHADER_NAME}.hlsl"
                VERBATIM
            )

            list(APPEND COMPILED_SHADERS "${SHADER_OUTPUT}")
        endforeach()

        add_custom_target(CompileShaders ALL DEPENDS ${COMPILED_SHADERS})
        add_dependencies(${PROJECT_NAME} CompileShaders)

        # Copy shaders to output directory after build
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${SHADER_OUTPUT_DIR}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/DirectX_Assets"
            COMMENT "Copying compiled shaders to output directory"
        )
    else()
        message(WARNING "DXC (DirectX Shader Compiler) not found. Shaders will not be compiled.")
    endif()

endif()

# ============================================================================
# macOS Configuration (Future Support)
# ============================================================================

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Build architectures")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum macOS version")

    target_compile_definitions(${PROJECT_NAME} PRIVATE
        MAC_ENV
        HAS_DIRECTX=0
        HAS_METAL=1
    )

    find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)

    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${COCOA_FRAMEWORK}
        ${METAL_FRAMEWORK}
    )

    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUNDLE TRUE
        BUNDLE_EXTENSION "plugin"
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_SOURCE_DIR}/resources/Info.plist"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

# Default install path for After Effects plugins
if(WIN32)
    set(AE_PLUGIN_DIR "C:/Program Files/Adobe/Adobe After Effects 2024/Support Files/Plug-ins/Effects" CACHE PATH "After Effects plugin directory")
else()
    set(AE_PLUGIN_DIR "/Applications/Adobe After Effects 2024/Plug-ins/Effects" CACHE PATH "After Effects plugin directory")
endif()

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${AE_PLUGIN_DIR}"
    LIBRARY DESTINATION "${AE_PLUGIN_DIR}"
    BUNDLE DESTINATION "${AE_PLUGIN_DIR}"
)

if(WIN32)
    install(DIRECTORY "${SHADER_OUTPUT_DIR}/"
        DESTINATION "${AE_PLUGIN_DIR}/DirectX_Assets"
        FILES_MATCHING PATTERN "*.cso"
    )
endif()

# ============================================================================
# Print Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== JustGlow Build Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "AE SDK Path: ${AE_SDK_PATH}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
if(WIN32)
    message(STATUS "DXC Compiler: ${DXC_EXECUTABLE}")
    message(STATUS "PiPL Tool: ${PIPL_TOOL}")
endif()
message(STATUS "Install Path: ${AE_PLUGIN_DIR}")
message(STATUS "====================================")
message(STATUS "")
