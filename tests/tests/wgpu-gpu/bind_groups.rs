use std::{array::from_fn, num::NonZeroU64};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages,
    ColorTargetState, ColorWrites, ComputePassDescriptor, ComputePipelineDescriptor,
    DownlevelFlags, Extent3d, FilterMode, FragmentState, Limits, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureViewDimension, VertexState,
};
use wgpu_test::{
    gpu_test, image::ReadbackBuffers, FailureCase, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

const UNUSED_VERTEX_SLOT: u32 = 15;
const EXTRA_VERTEX_UNIFORMS: u32 = 13;

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        MULTIPLE_BINDINGS_WITH_DIFFERENT_SIZES,
        BIND_GROUP_NONFILTERING_LAYOUT_NONFILTERING_SAMPLER,
        BIND_GROUP_NONFILTERING_LAYOUT_MIN_SAMPLER,
        BIND_GROUP_NONFILTERING_LAYOUT_MAG_SAMPLER,
        BIND_GROUP_NONFILTERING_LAYOUT_MIPMAP_SAMPLER,
        BIND_GROUP_WITH_MAX_BINDING_INDEX,
        STORAGE_ARRAY_LENGTHS_ACROSS_STAGES_AND_PIPELINES,
        RENDER_BIND_GROUP_REBINDING,
        RESOURCE_BINDINGS_ACROSS_STAGES_AND_PASSES,
    ]);
}

/// Create two bind groups against the same bind group layout, in the same
/// compute pass, but against two different shaders that have different binding
/// sizes. The first has binding size 8, the second has binding size 4.
///
/// Regression test for https://github.com/gfx-rs/wgpu/issues/7359.
fn multiple_bindings_with_differing_sizes(ctx: TestingContext) {
    const SHADER_SRC: &[&str] = &[
        "
        @group(0) @binding(0)
        var<uniform> buffer : vec2<f32>;

        @compute @workgroup_size(1, 1, 1) fn main() {
            // Just need a static use.
            let _value = buffer.x;
        }
        ",
        "
        @group(0) @binding(0)
        var<uniform> buffer : f32;

        @compute @workgroup_size(1, 1, 1) fn main() {
            // Just need a static use.
            let _value = buffer;
        }
        ",
    ];

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("buffer"),
        size: 8,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    let pipelines = SHADER_SRC
        .iter()
        .enumerate()
        .map(|(i, &shader_src)| {
            let module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("shader{i}")),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });

            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("pipeline{i}")),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        })
        .collect::<Vec<_>>();

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

    for (i, pipeline) in pipelines.iter().enumerate() {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bg{i}")),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: Some(NonZeroU64::new(u64::try_from(8 - 4 * i).unwrap()).unwrap()),
                }),
            }],
        });

        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[0]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    drop(cpass);

    let data = [0u8; 8];
    ctx.queue.write_buffer(&buffer, 0, &data);
    ctx.queue.submit(Some(encoder.finish()));

    ctx.device.poll(PollType::wait_indefinitely()).unwrap();
}

/// Test `descriptor` against a bind group layout that requires non-filtering sampler.
fn try_sampler_nonfiltering_layout(
    ctx: TestingContext,
    descriptor: &wgpu::SamplerDescriptor,
    good: bool,
) {
    let label = descriptor.label;
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            }],
        });

    let sampler = ctx.device.create_sampler(descriptor);

    let create_bind_group = || {
        let _ = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&sampler),
            }],
        });
    };

    if good {
        wgpu_test::valid(&ctx.device, create_bind_group);
    } else {
        wgpu_test::fail(
            &ctx.device,
            create_bind_group,
            Some("but given a sampler with filtering"),
        );
    }
}

#[gpu_test]
static MULTIPLE_BINDINGS_WITH_DIFFERENT_SIZES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(wgpu::Limits::downlevel_defaults())
            .enable_noop(),
    )
    .run_sync(multiple_bindings_with_differing_sizes);

#[gpu_test]
static BIND_GROUP_NONFILTERING_LAYOUT_NONFILTERING_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_nonfiltering_sampler"),
                    min_filter: wgpu::FilterMode::Nearest,
                    mag_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..wgpu::SamplerDescriptor::default()
                },
                true,
            );
        });

#[gpu_test]
static BIND_GROUP_NONFILTERING_LAYOUT_MIN_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_min_sampler"),
                    min_filter: wgpu::FilterMode::Linear,
                    mag_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..wgpu::SamplerDescriptor::default()
                },
                false,
            );
        });

#[gpu_test]
static BIND_GROUP_NONFILTERING_LAYOUT_MAG_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_mag_sampler"),
                    min_filter: wgpu::FilterMode::Nearest,
                    mag_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..wgpu::SamplerDescriptor::default()
                },
                false,
            );
        });

#[gpu_test]
static BIND_GROUP_NONFILTERING_LAYOUT_MIPMAP_SAMPLER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().enable_noop())
        .run_sync(|ctx| {
            try_sampler_nonfiltering_layout(
                ctx,
                &wgpu::SamplerDescriptor {
                    label: Some("bind_group_non_filtering_layout_mipmap_sampler"),
                    min_filter: wgpu::FilterMode::Nearest,
                    mag_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::MipmapFilterMode::Linear,
                    ..wgpu::SamplerDescriptor::default()
                },
                false,
            );
        });

#[gpu_test]
static BIND_GROUP_WITH_MAX_BINDING_INDEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(wgpu::Limits::downlevel_defaults())
            .expect_fail(FailureCase::kosmic_krisp()), // https://github.com/gfx-rs/wgpu/issues/9187
    )
    .run_async(|ctx| async move {
        let (device, queue) = ctx
            .adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_bindings_per_bind_group: ctx.adapter.limits().max_bindings_per_bind_group,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .unwrap();

        let max_binding_index = device.limits().max_bindings_per_bind_group - 1;
        let src_binding_index = max_binding_index - 1;
        let dst_binding_index = max_binding_index;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: src_binding_index,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: dst_binding_index,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let shader = format!(
            "
            @group(0) @binding({src_binding_index}) var<uniform> src: u32;
            @group(0) @binding({dst_binding_index}) var<storage, read_write> dst: u32;
            @compute @workgroup_size(1)
            fn main() {{
                dst = src;
            }}"
        );

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            module: &module,
            cache: None,
        });

        let test_value = 123u32;

        let src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            usage: wgpu::BufferUsages::UNIFORM,
            contents: &test_value.to_le_bytes(),
        });
        let dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: src_binding_index,
                    resource: wgpu::BindingResource::Buffer(src.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: dst_binding_index,
                    resource: wgpu::BindingResource::Buffer(dst.as_entire_buffer_binding()),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_bind_group(0, &bg, &[]);
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&dst, 0, &readback, 0, 4);
        queue.submit(Some(encoder.finish()));

        readback.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        assert_eq!(
            &*readback.slice(..).get_mapped_range().unwrap(),
            &test_value.to_le_bytes()
        );
    });
#[gpu_test]
static STORAGE_ARRAY_LENGTHS_ACROSS_STAGES_AND_PIPELINES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .downlevel_flags(wgpu::DownlevelFlags::VERTEX_STORAGE)
                .limits(wgpu::Limits {
                    max_vertex_buffers: UNUSED_VERTEX_SLOT + 1,
                    max_uniform_buffers_per_shader_stage: EXTRA_VERTEX_UNIFORMS,
                    ..Default::default()
                }),
        )
        .run_async(storage_array_lengths_across_stages_and_pipelines);

async fn storage_array_lengths_across_stages_and_pipelines(ctx: TestingContext) {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                r#"
            @group(0) @binding(0) var<storage, read> first: array<u32>;
            @group(0) @binding(1) var<storage, read> second: array<u32>;

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) @interpolate(flat) length: u32,
            }

            fn vertex_output(index: u32, length: u32) -> VertexOutput {
                let positions = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
                return VertexOutput(vec4f(positions[index], 0.0, 1.0), length);
            }

            @vertex fn vertex_first(@builtin(vertex_index) index: u32) -> VertexOutput {
                return vertex_output(index, arrayLength(&first));
            }
            @vertex fn vertex_second(@builtin(vertex_index) index: u32) -> VertexOutput {
                return vertex_output(index, arrayLength(&second));
            }
            @fragment fn fragment_first(input: VertexOutput) -> @location(0) vec4u {
                return vec4u(input.length, arrayLength(&first), first[0], 1u);
            }
            @fragment fn fragment_second(input: VertexOutput) -> @location(0) vec4u {
                return vec4u(input.length, arrayLength(&second), second[0], 1u);
            }
            "#
                .into(),
            ),
        });
    // Fifteen VS bindings place Metal's sizes table at the unused vertex slot.
    let entries: Vec<_> = (0..2 + EXTRA_VERTEX_UNIFORMS)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: if binding < 2 {
                wgpu::ShaderStages::VERTEX_FRAGMENT
            } else {
                wgpu::ShaderStages::VERTEX
            },
            ty: wgpu::BindingType::Buffer {
                ty: if binding < 2 {
                    wgpu::BufferBindingType::Storage { read_only: true }
                } else {
                    wgpu::BufferBindingType::Uniform
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    let pipelines = [
        ("vertex_first", "fragment_first"),
        ("vertex_second", "fragment_second"),
    ]
    .map(|(vertex, fragment)| {
        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fragment),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
    });
    let inputs = [&[17u32][..], &[29, 31, 37], &[41, 43, 47], &[53, 59]].map(|contents| {
        ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(contents),
            usage: BufferUsages::STORAGE,
        })
    });
    let unused_buffers = [0xffff_fffcu32, 0xffff_fff8].map(|value| {
        ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&value),
            usage: BufferUsages::VERTEX | BufferUsages::UNIFORM,
        })
    });
    let groups = [0, 1, 2].map(|first| {
        let entries: Vec<_> = (0..2 + EXTRA_VERTEX_UNIFORMS)
            .map(|binding| {
                let buffer = match binding {
                    0 => &inputs[first],
                    1 => &inputs[3],
                    _ => &unused_buffers[0],
                };
                wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                }
            })
            .collect();
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        })
    });
    let output = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 5,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Uint,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &attachments,
            ..Default::default()
        });
        for (pixel, (pipeline, group)) in [(0, 0), (0, 1), (0, 2), (1, 2), (0, 2)]
            .into_iter()
            .enumerate()
        {
            pass.set_viewport(pixel as f32, 0.0, 1.0, 1.0, 0.0, 1.0);
            pass.set_pipeline(&pipelines[pipeline]);
            pass.set_bind_group(0, &groups[group], &[]);
            pass.set_vertex_buffer(UNUSED_VERTEX_SLOT, unused_buffers[pixel % 2].slice(..));
            pass.draw(0..3, 0..1);
        }
    }
    let readback = ReadbackBuffers::new(&ctx.device, &output);
    readback.copy_from(&ctx.device, &mut encoder, &output);
    ctx.queue.submit(Some(encoder.finish()));
    let expected: [[u32; 4]; 5] = [
        [1, 1, 17, 1],
        [3, 3, 29, 1],
        [3, 3, 41, 1],
        [2, 2, 53, 1],
        [3, 3, 41, 1],
    ];
    readback
        .assert_buffer_contents(&ctx, bytemuck::cast_slice(&expected))
        .await;
}
const RENDER_REBIND_SHADER: &str = "
    @group(0) @binding(0) var<uniform> color: vec4f;
    @group(0) @binding(1) var image: texture_2d<f32>;

    @vertex
    fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
        let positions = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
        return vec4f(positions[index], 0.0, 1.0);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
        return color + textureLoad(image, vec2i(0), 0);
    }
";

fn render_rebind_pipeline(device: &wgpu::Device) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&group_layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(RENDER_REBIND_SHADER.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, group_layout)
}

async fn render_bind_group_rebinding(ctx: TestingContext) {
    let (pipeline, layout) = render_rebind_pipeline(&ctx.device);
    let input = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("partially initialized dynamic uniform"),
        size: 528,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&input, 0, bytemuck::cast_slice(&[1.0_f32, 0.0, 0.0, 1.0]));
    ctx.queue
        .write_buffer(&input, 256, bytemuck::cast_slice(&[0.0_f32, 0.0, 1.0, 1.0]));
    let textures = ["uninitialized then discarded", "initialized green"].map(|label| {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 2,
            },
            mip_level_count: 2,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    });
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &textures[1],
            mip_level: 1,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            aspect: wgpu::TextureAspect::All,
        },
        &[0, 255, 0, 0],
        wgpu::TexelCopyBufferLayout::default(),
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let views = textures.each_ref().map(|texture| {
        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_mip_level: 1,
            mip_level_count: Some(1),
            base_array_layer: 1,
            array_layer_count: Some(1),
            ..Default::default()
        })
    });
    let groups = views.each_ref().map(|view| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &input,
                        offset: 0,
                        size: NonZeroU64::new(16),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
            ],
        })
    });
    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rebound colors before and after discard"),
        size: wgpu::Extent3d {
            width: 4,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());
    let readback = ReadbackBuffers::new(&ctx.device, &target);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    for row in 0..2 {
        if row == 1 {
            let _discard = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &views[0],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                ..Default::default()
            });
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        for (column, (group, offset)) in [
            (&groups[0], 0),
            (&groups[1], 0),
            (&groups[0], 256),
            (&groups[0], 512),
        ]
        .into_iter()
        .enumerate()
        {
            pass.set_scissor_rect(column as u32, row, 1, 1);
            pass.set_bind_group(0, group, &[offset]);
            pass.draw(0..3, 0..1);
        }
    }
    readback.copy_from(&ctx.device, &mut encoder, &target);
    drop((groups, views, textures, input, pipeline));
    ctx.queue.submit([encoder.finish()]);
    readback
        .assert_buffer_contents(
            &ctx,
            &[
                255, 0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 0, 255, 0, 0, 255, 255,
                255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 0,
            ],
        )
        .await;
}

#[gpu_test]
static RENDER_BIND_GROUP_REBINDING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().limits(wgpu::Limits::downlevel_defaults()))
    .run_async(render_bind_group_rebinding);

#[gpu_test]
static RESOURCE_BINDINGS_ACROSS_STAGES_AND_PASSES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .limits(Limits::downlevel_defaults())
                .downlevel_flags(DownlevelFlags::VERTEX_STORAGE | DownlevelFlags::COMPUTE_SHADERS),
        )
        .run_async(resource_bindings_across_stages_and_passes);

async fn resource_bindings_across_stages_and_passes(ctx: TestingContext) {
    let device = &ctx.device;
    let alignment = device.limits().min_storage_buffer_offset_alignment as usize;
    let inputs = [[17u32, 41], [67, 89]].map(|values| {
        let mut bytes = vec![0; alignment + 8];
        bytes[..4].copy_from_slice(&values[0].to_le_bytes());
        bytes[alignment..alignment + 4].copy_from_slice(&values[1].to_le_bytes());
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some("binding inputs"),
            contents: &bytes,
            usage: BufferUsages::STORAGE,
        })
    });
    let textures = [
        [32, 0, 0, 255, 224, 0, 0, 255],
        [64, 0, 0, 255, 192, 0, 0, 255],
    ]
    .map(|pixels| {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("binding texture"),
            size: Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            texture.as_image_copy(),
            &pixels,
            TexelCopyBufferLayout {
                bytes_per_row: Some(8),
                ..Default::default()
            },
            texture.size(),
        );
        texture.create_view(&Default::default())
    });
    let samplers = [FilterMode::Nearest, FilterMode::Linear].map(|filter| {
        device.create_sampler(&SamplerDescriptor {
            min_filter: filter,
            mag_filter: filter,
            ..Default::default()
        })
    });
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let groups =
        [(0, 0, 0, 4), (1, 0, 1, 8), (1, 1, 0, 4)].map(|(buffer, texture, sampler, size)| {
            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &inputs[buffer],
                            offset: 0,
                            size: NonZeroU64::new(size),
                        }),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&textures[texture]),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Sampler(&samplers[sampler]),
                    },
                ],
            })
        });
    let output = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let output_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &output_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout), Some(&output_layout)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: None, source: ShaderSource::Wgsl(r#"
            @group(0) @binding(0) var<storage, read> values: array<u32>;
            @group(0) @binding(1) var image: texture_2d<f32>;
            @group(0) @binding(2) var image_sampler: sampler;
            @group(1) @binding(0) var<storage, read_write> output: array<u32>;
            override record: u32 = 0u;
            override bias: u32 = 0u;
            fn sample_value() -> u32 {
                return values[0] * arrayLength(&values) + u32(round(textureSampleLevel(image, image_sampler, vec2(0.5), 0.0).r * 255.0));
            }
            @compute @workgroup_size(1) fn compute() { output[record] = sample_value(); }
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) @interpolate(flat) value: u32,
            }
            @vertex fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput {
                let positions = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
                return VertexOutput(vec4(positions[index], 0.0, 1.0), sample_value());
            }
            @fragment fn fragment(input: VertexOutput) -> @location(0) u32 {
                return input.value + sample_value() + bias;
            }
        "#.into()),
    });
    let compute_pipelines: [_; 4] = from_fn(|record| {
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute"),
            compilation_options: PipelineCompilationOptions {
                constants: &[("record", record as f64)],
                ..Default::default()
            },
            cache: None,
        })
    });
    let render_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let render_pipelines = [0.0, 1.0].map(|bias| {
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: PipelineCompilationOptions {
                    constants: &[("bias", bias)],
                    ..Default::default()
                },
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::R32Uint,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    });
    let target = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 4,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Uint,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());
    let readback = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 512,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let unused_vertices = [4, 8].map(|size| {
        device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::VERTEX,
            mapped_at_creation: false,
        })
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let bindings = [(0, 0), (0, alignment as u32), (1, 0), (2, 0)];
    for _ in 0..2 {
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_bind_group(1, &output_group, &[]);
            for (index, &(group, offset)) in bindings.iter().enumerate() {
                pass.set_pipeline(&compute_pipelines[index]);
                pass.set_bind_group(0, &groups[group], &[offset]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: Default::default(),
            })],
            ..Default::default()
        });
        for (index, &(group, offset)) in bindings.iter().enumerate() {
            pass.set_pipeline(&render_pipelines[index % 2]);
            pass.set_bind_group(0, &groups[group], &[offset]);
            pass.set_vertex_buffer(0, unused_vertices[index % 2].slice(..));
            pass.set_viewport(index as f32, 0.0, 1.0, 1.0, 0.0, 1.0);
            pass.draw(0..3, 0..1);
        }
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 16);
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 256,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        target.size(),
    );
    ctx.queue.submit([encoder.finish()]);
    readback.slice(..).map_async(MapMode::Read, |result| {
        result.expect("resource binding results map after submission");
    });
    ctx.async_poll(PollType::wait_indefinitely())
        .await
        .expect("resource binding submission completes");
    let data = readback
        .slice(..)
        .get_mapped_range()
        .expect("resource binding readback is mapped");
    let words: &[u32] = bytemuck::cast_slice(&data);
    assert_eq!(words[..4], [241, 265, 262, 259]);
    assert_eq!(words[64..68], [482, 531, 524, 519]);
}
