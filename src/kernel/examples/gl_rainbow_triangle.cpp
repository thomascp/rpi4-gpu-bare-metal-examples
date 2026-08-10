#include <cstring>
#include "bit_utils.hpp"
#include "device/v3d/mbox.hpp"
#include "device/v3d/qpu_instr.hpp"
#include "device/v3d/v3d_cl.hpp"
#include "device/v3d/v3d_memory.hpp"
#include "device/v3d/v3d_qpu.hpp"
#include "kernel/cl_emitter.hpp"
#include "kernel/kernel.hpp"
#include "kernel/logging.hpp"

static void v3dInvalidateL2T()
{
    static constexpr u32 V3D_L2TCACTL_L2TFLS = bits::bit<u32>(0);
    static constexpr u32 V3D_L2TCACTL_FLM_FLUSH = bits::shiftL<u32>(0, 2, 1);
    ;

    mmio::write(mmio::V3D_CTL_L2TFLSTA, 0);
    mmio::write(mmio::V3D_CTL_L2TFLEND, ~0);
    mmio::write(mmio::V3D_CTL_L2TCACTL,
                V3D_L2TCACTL_L2TFLS | V3D_L2TCACTL_FLM_FLUSH);
}
static void v3dInvalidateSlices()
{
    mmio::write(mmio::V3D_CTL_SLCACTL, ~0);
}

static void v3dInvalidateCaches()
{
    v3dInvalidateL2T();
    v3dInvalidateSlices();
}

static size_t getBfCount()
{
    return mmio::read(mmio::V3D_CLE_BFC) & 0xFF;
}

static size_t getRfCount()
{
    return mmio::read(mmio::V3D_CLE_RFC) & 0xFF;
}

static constexpr u32 divRoundUp(u32 a, u32 b)
{
    return (a + b - 1) / b;
}

static void getSuperTiles(u32& supertile_w, u32& supertile_h,
                          u32& frame_w_supertile, u32& frame_h_supertile,
                          u32 tiles_x, u32 tiles_y)
{
    static constexpr u32 max_supertiles = 256;

    supertile_w = 1;
    supertile_h = 1;

    while (true)
    {
        frame_w_supertile = divRoundUp(tiles_x, supertile_w);
        frame_h_supertile = divRoundUp(tiles_y, supertile_h);

        if (frame_w_supertile * frame_h_supertile < max_supertiles)
            break;

        if (supertile_w < supertile_h)
            supertile_w++;
        else
            supertile_h++;
    }
}

/*
#version 300 es
precision highp float;

in vec4 v_color;

out vec4 outColor;

void main()
{
    outColor = v_color;
}
*/

static v3d::QPUInstr g_frag_shader_buff[] = {
0x3d103186bb800000, // nop                           ; nop                         ; ldvary.r0
0x5510b046bbc00000, // nop                           ; fmul r1, r0, rf0            ; ldvary.r2
0x551120c305ca9000, // fadd rf3, r1, r5              ; fmul r3, r2, rf0            ; ldvary.r4
0x5530600405d2b000, // fadd rf4, r3, r5              ; fmul r0, r4, rf0            ; thrsw; ldvary.r1
0x5420208505c68000, // fadd rf5, r0, r5              ; fmul r2, r1, rf0            ; thrsw
0x3c0021860582a000, // fadd rf6, r2, r5              ; nop
0x3c2031873583e0c4, // vfpack tlb, rf3, rf4          ; nop                         ; thrsw
0x3c0031873583e146, // vfpack tlb, rf5, rf6          ; nop
0x3c003186bb800000, // nop                           ; nop
};

/*
#version 300 es
in vec3 position;
in vec4 color;

out vec4 v_color;

void main()
{
    gl_Position = vec4(position, 1.0);
    v_color=color;
}
*/

static v3d::QPUInstr g_vtx_shader_buff[] = {
0x3de02183bc807000, // ldvpmv_in rf3, 0              ; nop
0x3de02184bc807001, // ldvpmv_in rf4, 1              ; nop
0x3de02185bc807002, // ldvpmv_in rf5, 2              ; nop
0x3de02186bc807003, // ldvpmv_in rf6, 3              ; nop
0x3de02187bc807004, // ldvpmv_in rf7, 4              ; nop
0x3c403186bb800000, // nop                           ; nop                         ; ldunif (vp_x_scale)
0x55e02008bcb870c5, // ldvpmv_in rf8, 5              ; fmul r0, rf3, r5
0x3de02189bc807006, // ldvpmv_in rf9, 6              ; nop
0x3c403181f6800000, // ffloor r1, r0                 ; nop                         ; ldunif (vp_y_scale)
0x3de02180f8837184, // stvpmv 4, rf6                 ; nop
0x54403083f5bb9100, // ftoiz r3, r1                  ; fmul r2, rf4, r5            ; ldunif (vp_z_scale)
0x3de02180f88371c5, // stvpmv 5, rf7                 ; nop
0x3de02180f8837206, // stvpmv 6, rf8                 ; nop
0x54403004f6b82140, // ffloor r4, r2                 ; fmul r0, rf5, r5            ; ldunif (vp_z_offset)
0x3de02180f8837247, // stvpmv 7, rf9                 ; nop
0x3c003181f583c000, // ftoiz r1, r4                  ; nop
0x3de02180f881f000, // stvpmv 0, r3                  ; nop
0x3c40318205828000, // fadd r2, r0, r5               ; nop                         ; ldunif (0x3f800000 / 1.000000)
0x3de02180f880f001, // stvpmv 1, r1                  ; nop
0x3de02180f8817002, // stvpmv 2, r2                  ; nop
0x3de02180f882f003, // stvpmv 3, r5                  ; nop
0x3c003186bb816000, // vpmwt -                       ; nop
0x3c203186bb800000, // nop                           ; nop                         ; thrsw
0x3c003186bb800000, // nop                           ; nop
0x3c003186bb800000, // nop                           ; nop
};

static v3d::QPUInstr g_coord_shader_buff[] = {
0x3de02183bc807000, // ldvpmv_in rf3, 0              ; nop
0x3c403186bb800000, // nop                           ; nop                         ; ldunif (vp_x_scale)
0x55e02004bcb870c1, // ldvpmv_in rf4, 1              ; fmul r0, rf3, r5
0x3de02185bc807002, // ldvpmv_in rf5, 2              ; nop
0x3c403181f6800000, // ffloor r1, r0                 ; nop                         ; ldunif (vp_y_scale)
0x3de02180f88370c0, // stvpmv 0, rf3                 ; nop
0x55e02080f8bb7101, // stvpmv 1, rf4                 ; fmul r2, rf4, r5
0x3c403183f5839000, // ftoiz r3, r1                  ; nop                         ; ldunif (0x3f800000 / 1.000000)
0x3de02180f8837142, // stvpmv 2, rf5                 ; nop
0x3c003184f6802000, // ffloor r4, r2                 ; nop
0x3de02180f882f003, // stvpmv 3, r5                  ; nop
0x3c003180f583c000, // ftoiz r0, r4                  ; nop
0x3de02180f881f004, // stvpmv 4, r3                  ; nop
0x3de02180f8807005, // stvpmv 5, r0                  ; nop
0x3c003186bb816000, // vpmwt -                       ; nop
0x3c203186bb800000, // nop                           ; nop                         ; thrsw
0x3c003186bb800000, // nop                           ; nop
0x3c003186bb800000, // nop                           ; nop
};

static f32 g_pos_attr_buff[] = {
    -1.f, -1.f, 0.f, 1.f, -1.f, 0.f, 0.f, 1.f, 0.f,
};

static u8 g_color_attr_buff[] = {
    0xFF, 0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0xFF,
};

void rainbowTriangleGL(Kernel& kern)
{
    size_t width = 500;
    size_t height = 500;

    CLEmitter bcl;
    CLEmitter rcl;
    CLEmitter indirect;
    CLEmitter attr0;
    CLEmitter attr1;
    CLEmitter state;

    /* =============== ATTRIBUTES ============== */

    u32 pos_attr_addr = attr0.paddr();
    attr0 << g_pos_attr_buff;
    u32 colorAttrAddr = attr1.paddr();
    attr1 << g_color_attr_buff;

    /* ================= STATE ================ */

    u32 default_attr_addr = state.paddr();

    static constexpr size_t V3D_MAX_VS_INPUTS = 64;

    for (size_t i = 0; i < V3D_MAX_VS_INPUTS / 4; i++)
        state << 0.0f << 0.0f << 0.0f << 1.0f;

    u32 frag_shader_addr = state.paddr();
    state << g_frag_shader_buff;
    u32 vtx_shader_addr = state.paddr();
    state << g_vtx_shader_buff;
    u32 coord_shader_addr = state.paddr();
    state << g_coord_shader_buff;

    /* =============== INDIRECT ============== */

    // fragment uniforms:
    u32 frag_unif_addr = indirect.paddr();

    // vertex uniforms
    u32 vtx_unif_addr = indirect.paddr();
    indirect << (width / 2) * 256.f;   // vp_x_scale
    indirect << (height / 2) * -256.f; // vp_y_scale
    indirect << 0.5f;                  // vp_z_scale
    indirect << 0.5f;                  // vp_z_offset
    indirect << 1.0f;                  // w

    // coordinate uniforms
    u32 coord_unif_addr = indirect.paddr();
    indirect << (width / 2) * 256.f;   // vp_x_scale
    indirect << (height / 2) * -256.f; // vp_y_scale
    indirect << 1.0f;                  // w

    // GL_SHADER_STATE_RECORD requires 32 byte alignment instead of 16
    // apparently?
    indirect.align(32);

    // GL_SHADER_STATE_RECORD
    u32 shader_state_record_addr = indirect.paddr();
    GLShaderStateRecord shader{};
    shader.enable_clipping = true;
    shader.fragment_shader_uses_real_pixel_centre_w_in_addition_to_centroid_w2 =
        true;
    shader.disable_implicit_point_line_varyings = true;
    shader.number_of_varyings_in_fragment_shader = 4; // 4 component color
    shader.coordinate_shader_output_vpm_segment_size = 1;
    shader.coordinate_shader_input_vpm_segment_size = 1;
    shader.vertex_shader_output_vpm_segment_size = 1;
    shader.vertex_shader_input_vpm_segment_size = 1;
    shader.address_of_default_attribute_values = default_attr_addr;

    shader.fragment_shader_code_address = frag_shader_addr >> 3;
    shader.fragment_shader_uniforms_address = frag_unif_addr;
    shader.fragment_shader_4_way_threadable = true;
    shader.fragment_shader_start_in_final_thread_section = false;
    shader.fragment_shader_propagate_nans = true;

    shader.vertex_shader_code_address = vtx_shader_addr >> 3;
    shader.vertex_shader_uniforms_address = vtx_unif_addr;
    shader.vertex_shader_4_way_threadable = true;
    shader.vertex_shader_start_in_final_thread_section = true;
    shader.vertex_shader_propagate_nans = true;

    shader.coordinate_shader_code_address = coord_shader_addr >> 3;
    shader.coordinate_shader_uniforms_address = coord_unif_addr;
    shader.coordinate_shader_4_way_threadable = true;
    shader.coordinate_shader_start_in_final_thread_section = true;
    shader.coordinate_shader_propagate_nans = true;
    indirect << shader;

    // GL_SHADER_ATTRIBUTE_RECORD
    size_t attr_count = 0;

    GlShaderStateAttributeRecord pos_attr{};
    pos_attr.address = pos_attr_addr;
    pos_attr.number_of_values_read_by_vertex_shader = 3;
    pos_attr.number_of_values_read_by_coordinate_shader = 3;
    pos_attr.instance_divisor = 0;
    pos_attr.stride = 3 * sizeof(f32);
    pos_attr.maximum_index = 0xFFFFFF;
    pos_attr.vec_size = 3;
    pos_attr.type = 2; // ATTRIBUTE_FLOAT
    indirect << pos_attr;
    attr_count++;

    GlShaderStateAttributeRecord color_attr;
    color_attr.address = colorAttrAddr;
    color_attr.number_of_values_read_by_vertex_shader = 4;
    color_attr.number_of_values_read_by_coordinate_shader = 0;
    color_attr.instance_divisor = 0;
    color_attr.stride = 4 * sizeof(u8);
    color_attr.maximum_index = 0xFFFFFF;
    color_attr.type = 4; // ATTRIBUTE_BYTE
    color_attr.normalized_int_type = true;
    indirect << color_attr;
    attr_count++;

    /* =============== RENDER SETUP ============== */

    V3DBuffer render_target(width * height * 4, 4096);
    std::fill_n(render_target.ptr<u32>(), width * height, 0xFFFFFFFF);
    // TODO: Z BUFFER * 2
    V3DBuffer z_buffer(width * height * 16, 4096);
    std::fill_n(render_target.ptr<u32>(), width * height, 0xFFFFFFFF);

    // decreases if msaa/double buffering/multiple color attachement/more bpp
    size_t tile_width = 64;
    size_t tile_height = 64;

    size_t vertex_count = sizeof(g_pos_attr_buff) / (sizeof(f32) * 3);
    size_t tilesX = divRoundUp(width, tile_width);
    size_t tilesY = divRoundUp(height, tile_height);

    size_t layer_count = 1;
    size_t tile_alloc_size = layer_count * tilesX * tilesY * 64;
    tile_alloc_size = bits::align<size_t>(tile_alloc_size, 4096);
    // first two chunks to avoid oom
    tile_alloc_size += 64 * 64 * 2;

    V3DBuffer tile_alloc(tile_alloc_size, 4096);

    size_t tsda_per_tile_size = 256; // 64 on Ver < 40
    V3DBuffer tile_state(layer_count * tilesY * tilesX * tsda_per_tile_size,
                         4096);

    /* =============== BCL ============== */
    {
        bcl << TileBinningModeCfg(0, 0, 1, 0, false, false, width, height);
        bcl << OP_FLUSH_VCD_CACHE;
        bcl << OcclusionQueryCounter(0);
        bcl << OP_START_TILE_BINNING;

        bcl << ClipWindow(0, 0, width, height);
        bcl << CfgBits(true, true, true, false, 0, 0, false, 7, false, false,
                       true, false, false, false, false);
        bcl << PointSize(1.0f);
        bcl << LineWidth(1.0f);
        bcl << ClipperXYScaling((width / 2) * 256.f, (height / 2) * -256.f);
        bcl << ClipperZScaleAndOffset(.5f, .5f);
        bcl << CLipperZMinMaxClippingPlanes(0.f, 1.f);
        bcl << ViewportOffset(width / 2, height / 2, 0, 0);
        bcl << ColorWriteMasks(0);
        bcl << BlendConstantColor(0, 0, 0, 0);
        bcl << OP_ZERO_ALL_FLAT_SHADE_FLAGS;
        bcl << OP_ZERO_ALL_NON_PERSPECTIVE_FLAGS;
        bcl << OP_ZERO_ALL_CENTROID_FLAGS;
        bcl << TransformFeedbackSpecs(0, false);
        bcl << OcclusionQueryCounter(0);
        bcl << SampleState(0xF, 1.0f);
        bcl << VcmCacheSize(4, 4);

        bcl << GlShaderState(shader_state_record_addr, attr_count);
        // mode=4 (triangle)
        bcl << VertexArrayPrims(4, vertex_count, 0);

        // bcl epilogue
        bcl << OP_FLUSH;
    }

    /* =============== RCL ============== */
    {
        // must be first
        rcl << TileRenderingModeCfgCommon(1, width, height, 0, false, false, 0,
                                          false, 2, false);
        rcl << TileRenderingModeCfgClearColorsPart1(0, 0xFF101010, 0x000000);
        rcl << TileRenderingModeCfgColor(0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        // must be last
        rcl << TileRenderingModeCfgZSClearValues(0, 1.0f);
        rcl << TileListInitialBlockSize(0, true);

        u32 supertile_w;
        u32 supertile_h;
        u32 frame_w_supertile;
        u32 frame_h_supertile;
        getSuperTiles(supertile_w, supertile_h, frame_w_supertile,
                      frame_h_supertile, tilesX, tilesY);
        // render layer
        {
            size_t layer_idx = 0;
            u32 tile_alloc_off = layer_idx * tilesX * tilesY * 64;
            rcl << MulticoreRenderingTileListSetBase(tile_alloc.paddr() +
                                                     tile_alloc_off);

            rcl << MulticoreRenderingSupertileCfg(
                supertile_w, supertile_h, frame_w_supertile, frame_h_supertile,
                tilesX, tilesY, 1, false, false);

            // clear
            rcl << TileCoordinates(0, 0);
            rcl << OP_END_OF_LOADS;
            rcl << StoreTileBufferGeneral();
            rcl << ClearTileBuffers(true, true);
            rcl << OP_END_OF_TILE_MARKER;
            // dummy store: workaround race condition
            rcl << TileCoordinates(0, 0);
            rcl << OP_END_OF_LOADS;
            rcl << StoreTileBufferGeneral();
            rcl << OP_END_OF_TILE_MARKER;

            rcl << OP_FLUSH_VCD_CACHE;

            // ****** INDIRECT ******
            {
                u32 indirect_addr = indirect.paddr();

                indirect << OP_TILE_COORDINATES_IMPLICIT;
                indirect << OP_END_OF_LOADS;

                indirect << PrimListFormat(LIST_TRIANGLES, false);
                // indirect << SetInstanceId(0); // ?
                indirect << BranchToImplicitTileList(0);

                indirect << StoreTileBufferGeneral(
                    BUFFER_RENDER_TARGET_0, V3D_TILING_RASTER, false,
                    V3D_DITHER_MODE_NONE, V3D_DECIMATE_MODE_SAMPLE_0,
                    V3D_OUTPUT_IMAGE_FORMAT_RGBA8, false, false, false,
                    width * 4, 0, render_target.paddr());

                indirect << StoreTileBufferGeneral(
                    BUFFER_Z, V3D_TILING_UIF_XOR, false, V3D_DITHER_MODE_NONE,
                    V3D_DECIMATE_MODE_SAMPLE_0, V3D_OUTPUT_IMAGE_FORMAT_D16,
                    false, false, false, (height + 15) / (2 * 4), 0,
                    z_buffer.paddr());

                // old?
                indirect << ClearTileBuffers(true, true);

                indirect << OP_END_OF_TILE_MARKER;
                indirect << OP_RETURN_FROM_SUB_LIST;

                rcl << StartAddressOfGenericTileList(indirect_addr,
                                                     indirect.paddr());
            }
        }

        size_t max_supertile_x = (width - 1) / (tile_width * supertile_w);
        size_t max_supertile_y = (height - 1) / (tile_height * supertile_h);

        for (size_t y = 0; y <= max_supertile_y; y++)
            for (size_t x = 0; x <= max_supertile_x; x++)
                rcl << SupertileCoorinates(x, y);

        // rcl end
        rcl << OP_END_OF_RENDERING;
    }

    // LOG("Shaders:\n");
    // HEXDUMP(state.startPtr<u8>(), state.size());
    // LOG("\n");
    LOG("Binning Command List:\n");
    HEXDUMP(bcl.startPtr(), bcl.size());
    LOG("\n");
    LOG("Rendering Command List:\n");
    HEXDUMP(rcl.startPtr(), rcl.size());
    LOG("\n");
    LOG("Indirect Command List:\n");
    HEXDUMP(indirect.startPtr(), indirect.size());
    LOG("\n");

    LOG("Invalidating caches...\n");
    v3dInvalidateCaches();

    LOG("Binning...\n");
    size_t last_bfc = getBfCount();
    v3d::qpu::executeBin(bcl.startPaddr(), bcl.endPaddr(), tile_alloc.paddr(),
                         tile_alloc_size, tile_state.paddr());

    while (getBfCount() <= last_bfc)
        ;

    LOG("Invalidating caches...\n");
    v3dInvalidateCaches();

    LOG("Rendering...\n");
    size_t last_rfc = getRfCount();
    v3d::qpu::executeRender(rcl.startPaddr(), rcl.endPaddr());

    while (getRfCount() <= last_rfc)
        ;

    kern.gfx()->drawer()->drawTex(render_target.ptr(), 600, 200, width, height);

    LOG("Done!\n\n\n");

    // LOG("Tile State:\n");
    // HEXDUMP(tileAlloc.ptr(), tileAlloc.size());
}
