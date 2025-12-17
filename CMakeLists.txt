# ============================================================================
# JustGlow - After Effects GPU Glow Plugin
# CMake Build Configuration
# ============================================================================

cmake_minimum_required(VERSION 3.20)
project(JustGlow VERSION 1.0.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA is optional - only enable on Windows if toolkit is found
option(ENABLE_CUDA "Enable CUDA support" ON)

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

# CUDA sources added conditionally later

set(PLUGIN_HEADERS
    src/JustGlow.h
    src/JustGlowParams.h
    src/JustGlowGPURenderer.h
    src/JustGlowCUDARenderer.h
)

set(CUDA_SOURCES
    src/JustGlowKernels.cu
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
    # Find CUDA (optional)
    set(CUDA_FOUND FALSE)
    if(ENABLE_CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(CUDA_FOUND TRUE)
            enable_language(CUDA)
            set(CMAKE_CUDA_STANDARD 17)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
            message(STATUS "CUDA Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")
        else()
            message(STATUS "CUDA Toolkit not found - building without CUDA support")
        endif()
    else()
        message(STATUS "CUDA support disabled by option")
    endif()

    # Preprocessor definitions
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        MSWindows
        WIN32
        _WINDOWS
        UNICODE
        _UNICODE
        HAS_DIRECTX=1
        HAS_CUDA=$<BOOL:${CUDA_FOUND}>
        PLUGIN_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        PLUGIN_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        PLUGIN_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        PLUGIN_VERSION_STRING="${PROJECT_VERSION}"
    )

    # Add CUDA include directories if available
    if(CUDA_FOUND)
        target_include_directories(${PROJECT_NAME} PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_sources(${PROJECT_NAME} PRIVATE
            src/JustGlowCUDARenderer.cpp
        )
    endif()

    # Configure RC file with version info
    configure_file(
        "${CMAKE_SOURCE_DIR}/resources/Win/JustGlow.rc.in"
        "${CMAKE_BINARY_DIR}/JustGlow.rc"
        @ONLY
    )

    # Add resource and module definition files
    target_sources(${PROJECT_NAME} PRIVATE
        "${CMAKE_BINARY_DIR}/JustGlow.rc"
        resources/Win/JustGlow.def
    )

    # Set output extension to .aex with version in filename
    set_target_properties(${PROJECT_NAME} PROPERTIES
        SUFFIX ".aex"
        PREFIX ""
        OUTPUT_NAME "JustGlow_v${PROJECT_VERSION}"
    )

    # Link libraries
    target_link_libraries(${PROJECT_NAME} PRIVATE
        d3d12.lib
        dxgi.lib
        d3dcompiler.lib
    )

    # Link CUDA if available
    if(CUDA_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE
            CUDA::cuda_driver
        )
    endif()

    # Compiler flags
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W3             # Warning level 3
        /WX-            # Warnings not as errors (for development)
        /MP             # Multi-processor compilation
    )

    # ========================================================================
    # PiPL Resource Generation using PiPLtool.exe
    # ========================================================================

    # Path to PiPLtool.exe (included in tools/ folder)
    set(PIPL_TOOL "${CMAKE_SOURCE_DIR}/tools/PiPLtool.exe" CACHE FILEPATH "Path to PiPLtool.exe")

    set(PIPL_R_FILE "${CMAKE_SOURCE_DIR}/resources/JustGlowPiPL.r")
    set(PIPL_RR_FILE "${CMAKE_BINARY_DIR}/JustGlowPiPL.rr")
    set(PIPL_RRC_FILE "${CMAKE_BINARY_DIR}/JustGlowPiPL.rrc")
    set(PIPL_TEMP_RC "${CMAKE_BINARY_DIR}/JustGlowPiPL_temp.rc")

    # Step 1: Preprocess .r file with cl.exe
    add_custom_command(
        OUTPUT "${PIPL_RR_FILE}"
        COMMAND cl.exe /I "${AE_SDK_PATH}" /I "${AE_SDK_PATH}/Util" /D "AE_OS_WIN" /D "AE_PROC_INTELx64" /EP "${PIPL_R_FILE}" > "${PIPL_RR_FILE}"
        DEPENDS "${PIPL_R_FILE}"
        COMMENT "Preprocessing PiPL.r"
        VERBATIM
    )

    # Step 2: Convert with PiPLtool.exe
    add_custom_command(
        OUTPUT "${PIPL_RRC_FILE}"
        COMMAND "${PIPL_TOOL}" "${PIPL_RR_FILE}" "${PIPL_RRC_FILE}"
        DEPENDS "${PIPL_RR_FILE}"
        COMMENT "Running PiPLtool"
        VERBATIM
    )

    # Step 3: Final preprocessing
    add_custom_command(
        OUTPUT "${PIPL_TEMP_RC}"
        COMMAND cl.exe /D "MSWindows" /EP "${PIPL_RRC_FILE}" > "${PIPL_TEMP_RC}"
        DEPENDS "${PIPL_RRC_FILE}"
        COMMENT "Generating PiPL RC file"
        VERBATIM
    )

    add_custom_target(GeneratePiPL DEPENDS "${PIPL_TEMP_RC}")
    add_dependencies(${PROJECT_NAME} GeneratePiPL)

    message(STATUS "PiPL will be generated using PiPLtool: ${PIPL_TOOL}")

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

    # ========================================================================
    # CUDA PTX Compilation (only if CUDA found)
    # ========================================================================

    if(CUDA_FOUND)
        set(CUDA_OUTPUT_DIR "${CMAKE_BINARY_DIR}/CUDA_Assets")
        file(MAKE_DIRECTORY "${CUDA_OUTPUT_DIR}")

        # Compile CUDA kernel to PTX
        set(CUDA_KERNEL_SRC "${CMAKE_SOURCE_DIR}/src/JustGlowKernels.cu")
        set(CUDA_KERNEL_PTX "${CUDA_OUTPUT_DIR}/JustGlowKernels.ptx")

        add_custom_command(
            OUTPUT "${CUDA_KERNEL_PTX}"
            COMMAND ${CMAKE_CUDA_COMPILER}
                -ptx
                -arch=sm_50
                -o "${CUDA_KERNEL_PTX}"
                "${CUDA_KERNEL_SRC}"
            DEPENDS "${CUDA_KERNEL_SRC}"
            COMMENT "Compiling CUDA kernel to PTX: JustGlowKernels.cu"
            VERBATIM
        )

        add_custom_target(CompileCUDA ALL DEPENDS "${CUDA_KERNEL_PTX}")
        add_dependencies(${PROJECT_NAME} CompileCUDA)

        # Copy CUDA PTX to output directory after build
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CUDA_OUTPUT_DIR}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>/CUDA_Assets"
            COMMENT "Copying compiled CUDA PTX to output directory"
        )

        message(STATUS "CUDA PTX will be compiled: ${CUDA_KERNEL_PTX}")
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
    if(CUDA_FOUND)
        install(DIRECTORY "${CUDA_OUTPUT_DIR}/"
            DESTINATION "${AE_PLUGIN_DIR}/CUDA_Assets"
            FILES_MATCHING PATTERN "*.ptx"
        )
    endif()
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
    if(CUDA_FOUND)
        message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
        message(STATUS "NVCC: ${CMAKE_CUDA_COMPILER}")
    else()
        message(STATUS "CUDA: Not available")
    endif()
    message(STATUS "PiPL Tool: ${PIPL_TOOL}")
endif()
message(STATUS "Install Path: ${AE_PLUGIN_DIR}")
message(STATUS "====================================")
message(STATUS "")
