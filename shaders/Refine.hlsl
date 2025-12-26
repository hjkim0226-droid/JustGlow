/**
 * JustGlow Refine Shader
 *
 * Calculates BoundingBox of active pixels (L >= threshold)
 * and generates IndirectArgs for DispatchIndirect.
 *
 * Two passes:
 * 1. RefineCS: Atomic min/max to find bounds
 * 2. CalcIndirectArgsCS: Convert bounds to thread group counts
 */

#include "Common.hlsli"

#define THREAD_GROUP_SIZE 16

// ============================================================================
// Refine Constant Buffer
// ============================================================================

cbuffer RefineParams : register(b2)
{
    int     g_refineWidth;
    int     g_refineHeight;
    float   g_refineThreshold;
    int     g_blurRadius;

    int     g_mipLevel;         // Current MIP level (0 = prefilter)
    int     g_maxMipLevels;
    int     _refinePad0;
    int     _refinePad1;
};

// ============================================================================
// UAV Resources
// ============================================================================

// Atomic bounds buffer: [minX, maxX, minY, maxY]
// Initialized to [width, 0, height, 0] before dispatch
globallycoherent RWStructuredBuffer<uint> g_atomicBounds : register(u1);

// IndirectArgs buffer for DispatchIndirect
// Structure: { ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ }
RWStructuredBuffer<uint> g_indirectArgs : register(u2);

// Bounds output for next stage (read by Prefilter/Downsample)
RWStructuredBuffer<int> g_boundsOutput : register(u3);

// ============================================================================
// Pass 1: RefineCS - Find BoundingBox using atomics
// ============================================================================

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void RefineCS(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_refineWidth || dispatchID.y >= (uint)g_refineHeight)
        return;

    // Sample input texture
    float4 pixel = g_inputTex[dispatchID.xy];

    // Calculate luminance (Rec.601 weights, matches CUDA)
    float L = 0.299f * pixel.r + 0.587f * pixel.g + 0.114f * pixel.b;

    // If pixel is above threshold, expand bounding box
    if (L >= g_refineThreshold)
    {
        // Expand by blur radius to ensure blur kernel has full coverage
        int expandedMinX = max(0, (int)dispatchID.x - g_blurRadius);
        int expandedMaxX = min(g_refineWidth - 1, (int)dispatchID.x + g_blurRadius);
        int expandedMinY = max(0, (int)dispatchID.y - g_blurRadius);
        int expandedMaxY = min(g_refineHeight - 1, (int)dispatchID.y + g_blurRadius);

        // Atomic update of global bounds
        // Note: Using globallycoherent for cross-thread-group visibility
        InterlockedMin(g_atomicBounds[0], (uint)expandedMinX);
        InterlockedMax(g_atomicBounds[1], (uint)expandedMaxX);
        InterlockedMin(g_atomicBounds[2], (uint)expandedMinY);
        InterlockedMax(g_atomicBounds[3], (uint)expandedMaxY);
    }
}

// ============================================================================
// Pass 2: CalcIndirectArgsCS - Convert bounds to thread group counts
// ============================================================================

[numthreads(1, 1, 1)]
void CalcIndirectArgsCS(uint3 dispatchID : SV_DispatchThreadID)
{
    // Read atomic bounds
    uint minX = g_atomicBounds[0];
    uint maxX = g_atomicBounds[1];
    uint minY = g_atomicBounds[2];
    uint maxY = g_atomicBounds[3];

    // Check if valid bounds (at least one pixel above threshold)
    bool validBounds = (maxX >= minX) && (maxY >= minY);

    uint width, height;
    int outMinX, outMinY, outMaxX, outMaxY;

    if (validBounds)
    {
        width = maxX - minX + 1;
        height = maxY - minY + 1;
        outMinX = (int)minX;
        outMinY = (int)minY;
        outMaxX = (int)maxX;
        outMaxY = (int)maxY;
    }
    else
    {
        // No active pixels - use full image as fallback
        width = (uint)g_refineWidth;
        height = (uint)g_refineHeight;
        outMinX = 0;
        outMinY = 0;
        outMaxX = g_refineWidth - 1;
        outMaxY = g_refineHeight - 1;
    }

    // Calculate thread group counts (ceiling division)
    uint threadGroupCountX = (width + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
    uint threadGroupCountY = (height + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

    // Write IndirectArgs
    // Index = mipLevel * 3 (each level has 3 uints: X, Y, Z)
    uint baseIdx = (uint)g_mipLevel * 3;
    g_indirectArgs[baseIdx + 0] = threadGroupCountX;
    g_indirectArgs[baseIdx + 1] = threadGroupCountY;
    g_indirectArgs[baseIdx + 2] = 1;  // Z is always 1

    // Write bounds for next stage to read
    // Index = mipLevel * 4 (each level has 4 ints: minX, maxX, minY, maxY)
    uint boundsBaseIdx = (uint)g_mipLevel * 4;
    g_boundsOutput[boundsBaseIdx + 0] = outMinX;
    g_boundsOutput[boundsBaseIdx + 1] = outMaxX;
    g_boundsOutput[boundsBaseIdx + 2] = outMinY;
    g_boundsOutput[boundsBaseIdx + 3] = outMaxY;
}

// ============================================================================
// Pass 3: ResetBoundsCS - Reset atomic bounds for next frame
// ============================================================================

[numthreads(1, 1, 1)]
void ResetBoundsCS(uint3 dispatchID : SV_DispatchThreadID)
{
    // Reset to invalid state
    // minX/minY = max value (will be reduced by InterlockedMin)
    // maxX/maxY = 0 (will be increased by InterlockedMax)
    g_atomicBounds[0] = (uint)g_refineWidth;   // minX
    g_atomicBounds[1] = 0;                      // maxX
    g_atomicBounds[2] = (uint)g_refineHeight;  // minY
    g_atomicBounds[3] = 0;                      // maxY
}
