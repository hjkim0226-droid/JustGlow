/**
 * JustGlow PiPL Resource
 *
 * Plug-in Property List for After Effects
 * Defines plugin metadata and capabilities.
 */

#include "AEConfig.h"
#include "AE_EffectVers.h"

#ifndef AE_OS_WIN
    #include "AE_General.r"
#endif

resource 'PiPL' (16000) {
    {
        /* Basic plugin info */
        Kind {
            AEEffect
        },

        Name {
            "JustGlow"
        },

        Category {
            "Stylize"
        },

        /* Entry point */
#ifdef AE_OS_WIN
        CodeWin64X86 {
            "EffectMain"
        },
#else
        CodeMacARM64 {
            "EffectMain"
        },
        CodeMacIntel64 {
            "EffectMain"
        },
#endif

        /* Version info */
        AE_PiPL_Version {
            2,
            0
        },

        AE_Effect_Spec_Version {
            PF_PLUG_IN_VERSION,
            PF_PLUG_IN_SUBVERS
        },

        AE_Effect_Version {
            0x00080001  /* 1.0.0 - PF_VERSION(1,0,0,PF_Stage_DEVELOP,1) */
        },

        AE_Effect_Info_Flags {
            0
        },

        AE_Effect_Global_OutFlags {
            0x06000600  /* PF_OutFlag_DEEP_COLOR_AWARE (1<<25) |
                          PF_OutFlag_SEND_UPDATE_PARAMS_UI (1<<26) |
                          PF_OutFlag_PIX_INDEPENDENT (1<<10) |
                          PF_OutFlag_I_EXPAND_BUFFER (1<<9) */
        },

        AE_Effect_Global_OutFlags_2 {
            0x2A001400  /* PF_OutFlag2_SUPPORTS_SMART_RENDER (0x00000400) |
                          PF_OutFlag2_FLOAT_COLOR_AWARE (0x00001000) |
                          PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 (0x02000000) |
                          PF_OutFlag2_SUPPORTS_THREADED_RENDERING (0x08000000) |
                          PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING (0x20000000) */
            /* GPU rendering: CUDA via GPU_RENDER_F32, DirectX via DIRECTX_RENDERING */
        },

        AE_Effect_Match_Name {
            "com.justglow.effect"
        },

        AE_Reserved_Info {
            0
        },

        AE_Effect_Support_URL {
            "https://github.com/justglow"
        }
    }
};
