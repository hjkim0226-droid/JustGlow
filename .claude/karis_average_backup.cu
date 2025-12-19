/**
 * Karis Average Backup
 *
 * Removed from JustGlowKernels.cu on 2025-12-20
 * Reason: Causes artifacts on transparent backgrounds
 *         - Transparent pixels (luminance=0) get weight=1.0 (maximum)
 *         - This pollutes color average with transparent areas
 *         - Originally designed for real-time games (frame-to-frame flickering)
 *         - Not needed in After Effects (no temporal flickering)
 *
 * Keep for reference in case HDR anti-firefly is needed in future.
 */

// ============================================================================
// Karis Weight Function
// Returns weight inversely proportional to luminance
// Bright pixels get lower weight to prevent fireflies
// Problem: Transparent pixels (luma=0) get weight=1.0 (maximum!)
// ============================================================================

__device__ __forceinline__ float karisWeight(float r, float g, float b) {
    return 1.0f / (1.0f + luminance(r, g, b));
}

// ============================================================================
// Karis Average for 4 samples
// Weights each sample by inverse luminance, then normalizes
// ============================================================================

__device__ void karisAverage4(
    float r0, float g0, float b0,
    float r1, float g1, float b1,
    float r2, float g2, float b2,
    float r3, float g3, float b3,
    float& outR, float& outG, float& outB)
{
    float w0 = karisWeight(r0, g0, b0);
    float w1 = karisWeight(r1, g1, b1);
    float w2 = karisWeight(r2, g2, b2);
    float w3 = karisWeight(r3, g3, b3);

    float totalWeight = w0 + w1 + w2 + w3;

    outR = (r0 * w0 + r1 * w1 + r2 * w2 + r3 * w3) / totalWeight;
    outG = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3) / totalWeight;
    outB = (b0 * w0 + b1 * w1 + b2 * w2 + b3 * w3) / totalWeight;
}

// ============================================================================
// PrefilterKernel HDR Branch (Karis-weighted)
// ============================================================================

/*
    if (useHDR) {
        // Karis average with alpha weighting
        // Group 1: Inner corners (D, E, I, J) - weight 0.5
        float g1r, g1g, g1b;
        karisAverage4(Dr, Dg, Db, Er, Eg, Eb, Ir, Ig, Ib, Jr, Jg, Jb, g1r, g1g, g1b);
        float g1a = (Da + Ea + Ia + Ja) * 0.25f;

        // Group 2: Top-left (A, B, F, G) - weight 0.125
        float g2r, g2g, g2b;
        karisAverage4(Ar, Ag, Ab, Br, Bg, Bb, Fr, Fg, Fb, Gr, Gg, Gb, g2r, g2g, g2b);
        float g2a = (Aa + Ba + Fa + Ga) * 0.25f;

        // Group 3: Top-right (B, C, G, H) - weight 0.125
        float g3r, g3g, g3b;
        karisAverage4(Br, Bg, Bb, Cr, Cg, Cb, Gr, Gg, Gb, Hr, Hg, Hb, g3r, g3g, g3b);
        float g3a = (Ba + Ca + Ga + Ha) * 0.25f;

        // Group 4: Bottom-left (F, G, K, L) - weight 0.125
        float g4r, g4g, g4b;
        karisAverage4(Fr, Fg, Fb, Gr, Gg, Gb, Kr, Kg, Kb, Lr, Lg, Lb, g4r, g4g, g4b);
        float g4a = (Fa + Ga + Ka + La) * 0.25f;

        // Group 5: Bottom-right (G, H, L, M) - weight 0.125
        float g5r, g5g, g5b;
        karisAverage4(Gr, Gg, Gb, Hr, Hg, Hb, Lr, Lg, Lb, Mr, Mg, Mb, g5r, g5g, g5b);
        float g5a = (Ga + Ha + La + Ma) * 0.25f;

        // Weighted sum (RGB stays premultiplied through Karis)
        sumR = g1r * 0.5f + (g2r + g3r + g4r + g5r) * 0.125f;
        sumG = g1g * 0.5f + (g2g + g3g + g4g + g5g) * 0.125f;
        sumB = g1b * 0.5f + (g2b + g3b + g4b + g5b) * 0.125f;
        sumA = g1a * 0.5f + (g2a + g3a + g4a + g5a) * 0.125f;
    }
*/
