use std::num::NonZeroU32;

use wgpu::*;
use wgpu_test::{
    gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

#[gpu_test]
static REPEATED_VIEWS_PRESERVE_SUBRESOURCE_INITIALIZATION: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(
                    Features::TEXTURE_BINDING_ARRAY
                        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                )
                .limits(Limits {
                    max_binding_array_elements_per_shader_stage: 4,
                    ..Default::default()
                }),
        )
        .run_async(repeated_views_preserve_subresource_initialization);

async fn repeated_views_preserve_subresource_initialization(ctx: TestingContext) {
    let texture = ctx.device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 2,
        },
        mip_level_count: 2,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let views = [(0, 0), (0, 1), (1, 1)].map(|(mip, layer)| {
        texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2),
            base_mip_level: mip,
            mip_level_count: Some(1),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        })
    });
    for (mip, layer, texel) in [(0, 0, [11, 13, 17, 19]), (1, 1, [23, 29, 31, 37])] {
        let width = 2 >> mip;
        ctx.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip,
                origin: Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: TextureAspect::All,
            },
            &texel.repeat((width * width) as usize),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(width),
            },
            Extent3d {
                width,
                height: width,
                depth_or_array_layers: 1,
            },
        );
    }
    let layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: NonZeroU32::new(4),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let output = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureViewArray(&[
                    &views[0], &views[1], &views[0], &views[2],
                ]),
            },
            BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(
            r#"
            enable wgpu_binding_array;
            @group(0) @binding(0) var textures: binding_array<texture_2d<f32>>;
            @group(0) @binding(1) var<storage, read_write> output: array<vec4u>;
            @compute @workgroup_size(1)
            fn main(@builtin(global_invocation_id) index: vec3u) {
                output[index.x] = vec4u(round(textureLoad(textures[index.x], vec2u(0), 0) * 255.0));
            }
        "#
            .into(),
        ),
    });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    for (load, store) in [
        (LoadOp::Clear(Color::WHITE), StoreOp::Store),
        (LoadOp::Load, StoreOp::Discard),
    ] {
        let attachments = [Some(RenderPassColorAttachment {
            view: &views[1],
            depth_slice: None,
            resolve_target: None,
            ops: Operations { load, store },
        })];
        drop(encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &attachments,
            ..Default::default()
        }));
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 64);
    ctx.queue.submit(Some(encoder.finish()));
    readback.slice(..).map_async(MapMode::Read, |_| ());
    ctx.async_poll(PollType::wait_indefinitely())
        .await
        .expect("texture readback completes");
    let data = readback
        .slice(..)
        .get_mapped_range()
        .expect("texture readback is mapped");
    assert_eq!(
        bytemuck::cast_slice::<u8, u32>(&data),
        [11, 13, 17, 19, 0, 0, 0, 0, 11, 13, 17, 19, 23, 29, 31, 37],
    );
    drop(data);
    readback.unmap();
}

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        BINDING_ARRAY_SAMPLED_TEXTURES,
        PARTIAL_BINDING_ARRAY_SAMPLED_TEXTURES,
        PARTIAL_BINDING_ARRAY_FOLLOWED_BY_STORAGE_BUFFER,
        REPEATED_VIEWS_PRESERVE_SUBRESOURCE_INITIALIZATION,
    ]);
}

#[gpu_test]
static BINDING_ARRAY_SAMPLED_TEXTURES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .instance_flags(wgpu::InstanceFlags::GPU_BASED_VALIDATION)
            .features(
                Features::TEXTURE_BINDING_ARRAY
                    | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            )
            .limits(Limits {
                max_binding_array_elements_per_shader_stage: 16,
                ..Limits::default()
            })
            // https://github.com/gfx-rs/wgpu/issues/9184
            .expect_fail(
                wgpu_test::FailureCase::molten_vk()
                    .validation_error("Shader library compile failed")
                    .validation_error("could not be compiled into pipeline"),
            ),
    )
    .run_async(|ctx| async move { binding_array_sampled_textures(ctx, false).await });

#[gpu_test]
static PARTIAL_BINDING_ARRAY_SAMPLED_TEXTURES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .instance_flags(wgpu::InstanceFlags::GPU_BASED_VALIDATION)
            .features(
                Features::TEXTURE_BINDING_ARRAY
                    | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                    | Features::PARTIALLY_BOUND_BINDING_ARRAY,
            )
            .limits(Limits {
                max_binding_array_elements_per_shader_stage: 32,
                ..Limits::default()
            })
            // https://github.com/gfx-rs/wgpu/issues/9184
            .expect_fail(
                wgpu_test::FailureCase::molten_vk()
                    .validation_error("Shader library compile failed")
                    .validation_error("could not be compiled into pipeline"),
            ),
    )
    .run_async(|ctx| async move { binding_array_sampled_textures(ctx, false).await });

/// Test to see how texture bindings array work and additionally making sure
/// that non-uniform indexing is working correctly.
///
/// If non-uniform indexing is not working correctly, AMD will produce the wrong
/// output due to non-native support for non-uniform indexing within a WARP.
async fn binding_array_sampled_textures(ctx: TestingContext, partially_bound: bool) {
    let shader = r#"
        enable wgpu_binding_array;
        @group(0) @binding(0)
        var textures: binding_array<texture_2d<f32>>;

        @vertex
        fn vertMain(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(
                vec2f(-1.0, -1.0),
                vec2f(3.0, -1.0),
                vec2f(-1.0, 3.0)
            );

            return vec4<f32>(positions[id], 0.0, 1.0);
        }

        @fragment
        fn fragMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
            let pixel = vec2u(floor(pos.xy));
            let index = pixel.y * 4 + pixel.x;

            return textureLoad(textures[index], vec2u(0), 0);
        }
    "#;

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Binding Array Texture"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

    let image = image::load_from_memory(include_bytes!("../3x3_colors.png")).unwrap();
    // Resize image to 4x4
    let image = image
        .resize_exact(4, 4, image::imageops::FilterType::Gaussian)
        .into_rgba8();

    // Create one texture for each pixel
    let mut input_views = Vec::with_capacity(64);
    for data in image.pixels() {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        ctx.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &data.0,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        input_views.push(texture.create_view(&TextureViewDescriptor::default()));
    }

    let output_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Output Texture"),
        size: Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let output_view = output_texture.create_view(&TextureViewDescriptor::default());

    let count = if partially_bound { 32 } else { 16 };

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(NonZeroU32::new(count).unwrap()),
            }],
        });

    let input_view_references: Vec<_> = input_views.iter().collect();

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureViewArray(&input_view_references),
        }],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vertMain"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fragMain"),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &output_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    let readback_buffers = ReadbackBuffers::new(&ctx.device, &output_texture);
    readback_buffers.copy_from(&ctx.device, &mut encoder, &output_texture);

    ctx.queue.submit(Some(encoder.finish()));

    readback_buffers.assert_buffer_contents(&ctx, &image).await;
}

#[gpu_test]
static PARTIAL_BINDING_ARRAY_FOLLOWED_BY_STORAGE_BUFFER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .instance_flags(wgpu::InstanceFlags::GPU_BASED_VALIDATION)
                .features(
                    Features::TEXTURE_BINDING_ARRAY
                        | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                        | Features::PARTIALLY_BOUND_BINDING_ARRAY,
                )
                .limits(Limits {
                    max_binding_array_elements_per_shader_stage: 32,
                    ..Limits::default()
                }),
        )
        .run_async(
            |ctx| async move { partial_binding_array_followed_by_storage_buffer(ctx).await },
        );

/// Regression test for a DX12 descriptor-table bug. A *partially-bound* binding
/// array (binding 0) is followed by another binding (binding 1, a storage buffer)
/// in the same bind group. On DX12 the root-signature range reserves the array's
/// full declared size and naga assigns binding 1's shader register *after* the
/// whole array, but the backend used to stage only the bound array views —
/// shifting binding 1's descriptor so the buffer was read from the wrong heap slot
/// (garbage). The shader output here depends only on the trailing buffer, so a
/// misaligned descriptor makes the readback differ from the marker color.
async fn partial_binding_array_followed_by_storage_buffer(ctx: TestingContext) {
    let shader = r#"
        enable wgpu_binding_array;
        @group(0) @binding(0)
        var textures: binding_array<texture_2d<f32>>;
        @group(0) @binding(1)
        var<storage, read> marker: vec4<f32>;

        @vertex
        fn vertMain(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
            var positions = array<vec2f, 3>(
                vec2f(-1.0, -1.0),
                vec2f(3.0, -1.0),
                vec2f(-1.0, 3.0)
            );
            return vec4f(positions[id], 0.0, 1.0);
        }

        @fragment
        fn fragMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
            // Keep the binding array live (so it stays in the layout before the
            // buffer), but make the output depend only on the trailing buffer.
            let keep = textureLoad(textures[0], vec2u(0), 0).x * 0.0;
            return marker + vec4f(keep);
        }
    "#;

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Partial Array + Storage Buffer"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

    // A single input texture for the partially-bound array (1 of 32 slots).
    let input_texture = ctx.device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 1,
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
    let input_view = input_texture.create_view(&TextureViewDescriptor::default());

    // The marker the shader must echo. [0, 1, 0, 1] -> Rgba8Unorm [0, 255, 0, 255].
    let marker = [0.0f32, 1.0, 0.0, 1.0];
    let marker_bytes: Vec<u8> = marker.iter().flat_map(|f| f.to_le_bytes()).collect();
    let marker_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: marker_bytes.len() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&marker_buffer, 0, &marker_bytes);

    let output_texture = ctx.device.create_texture(&TextureDescriptor {
        label: Some("Output Texture"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let output_view = output_texture.create_view(&TextureViewDescriptor::default());

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: Some(NonZeroU32::new(32).unwrap()),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    // Bind only 1 view into the 32-slot array (partial), plus the buffer after it.
    let views = [&input_view];
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureViewArray(&views),
            },
            BindGroupEntry {
                binding: 1,
                resource: marker_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vertMain"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fragMain"),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &output_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::RED),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    let readback_buffers = ReadbackBuffers::new(&ctx.device, &output_texture);
    readback_buffers.copy_from(&ctx.device, &mut encoder, &output_texture);
    ctx.queue.submit(Some(encoder.finish()));

    // The output must equal the marker. Before the fix, the buffer's descriptor
    // was misaligned on DX12 and this read garbage.
    let expected: [u8; 4] = [0, 255, 0, 255];
    readback_buffers
        .assert_buffer_contents(&ctx, &expected)
        .await;
}
