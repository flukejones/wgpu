//! Tests for buffer copy validation.

use wgpu::PollType;
use wgpu_test::{apply, fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        QUEUE_WRITE_TEXTURE_THEN_DESTROY,
        QUEUE_WRITE_TEXTURE_OVERFLOW,
        QUEUE_WRITE_TEXTURE_BUFFER_OOB,
        QUEUE_WRITE_BUFFER_RECYCLING,
    ]);
}

#[apply(gpu_test!)]
static QUEUE_WRITE_BUFFER_RECYCLING: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let destination = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("queue write destination"),
            size: 300 * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("queue write snapshots"),
            size: destination.size() * 3,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut retained_buffers = None;
        for cycle in 0..2_u32 {
            for submission in 0..3_u32 {
                for slot in 0..300_u32 {
                    let data = [cycle, submission, slot, 0xdead_beef];
                    if slot % 2 == 0 {
                        ctx.queue.write_buffer(
                            &destination,
                            u64::from(slot) * 16,
                            bytemuck::cast_slice(&data),
                        );
                    } else {
                        ctx.queue
                            .write_buffer_with(
                                &destination,
                                u64::from(slot) * 16,
                                16.try_into().expect("nonempty upload"),
                            )
                            .expect("valid queue write")
                            .copy_from_slice(bytemuck::cast_slice(&data));
                    }
                    if cycle == 1 && submission == 0 && slot == 0 {
                        let buffers = ctx.device.get_internal_counters().hal.buffers.read();
                        if buffers != 0 {
                            assert_eq!(
                                Some(buffers),
                                retained_buffers,
                                "the warmed upload must reuse a completed native allocation"
                            );
                        }
                    }
                }
                let mut encoder = ctx.device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(
                    &destination,
                    0,
                    &readback,
                    u64::from(submission) * destination.size(),
                    destination.size(),
                );
                ctx.queue.submit([encoder.finish()]);
            }
            readback.slice(..).map_async(wgpu::MapMode::Read, |result| {
                result.expect("queue write readback mapping");
            });
            ctx.async_poll(PollType::wait_indefinitely())
                .await
                .expect("queue write completion");
            {
                let mapped = readback
                    .slice(..)
                    .get_mapped_range()
                    .expect("mapped snapshots");
                let words: &[u32] = bytemuck::cast_slice(&mapped);
                for submission in 0..3_u32 {
                    for slot in 0..300_u32 {
                        let offset = (submission * 300 + slot) as usize * 4;
                        assert_eq!(
                            &words[offset..offset + 4],
                            &[cycle, submission, slot, 0xdead_beef],
                            "queue writes must preserve each submitted snapshot"
                        );
                    }
                }
            }
            readback.unmap();
            retained_buffers = Some(ctx.device.get_internal_counters().hal.buffers.read());
        }
        ctx.queue.write_buffer(&destination, 0, &[9; 16]);
        destination.destroy();
        ctx.queue.submit([]);
        ctx.async_poll(PollType::wait_indefinitely())
            .await
            .expect("destroyed destination upload completion");
    });

#[apply(gpu_test!)]
static QUEUE_WRITE_TEXTURE_THEN_DESTROY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 64,
                height: 32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let data = vec![255; 1024];

        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1024),
                rows_per_image: Some(32),
            },
            wgpu::Extent3d {
                width: 64,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        // Unlike textures used in a command buffer, which must not be destroyed prior to calling
        // submit, it is permissible to destroy a texture used in an immediate queue operation
        // before calling submit.
        texture.destroy();

        ctx.queue.submit([]);
        ctx.device.poll(PollType::wait_indefinitely()).unwrap();
    });

#[apply(gpu_test!)]
static QUEUE_WRITE_TEXTURE_OVERFLOW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 146,
                height: 25,
                depth_or_array_layers: 192,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let data = vec![255; 128];

        fail(
            &ctx.device,
            || {
                ctx.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(879161360),
                        //bytes_per_image: 4294967295,
                        rows_per_image: Some(4294967295 / 879161360),
                    },
                    wgpu::Extent3d {
                        width: 3056263286,
                        height: 64,
                        depth_or_array_layers: 4294967295,
                    },
                );
            },
            Some("end up overrunning the bounds of the destination texture"),
        );
    });

#[apply(gpu_test!)]
static QUEUE_WRITE_TEXTURE_BUFFER_OOB: GpuTestConfiguration =
    GpuTestConfiguration::new().run_sync(|ctx| {
        // Test that transfers overrunning the end of the source buffer, or
        // where offset + size overflows a u64, are rejected.
        for offset in [120, u64::MAX - 3] {
            let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 146,
                    height: 25,
                    depth_or_array_layers: 192,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            let data = vec![255; 128];

            fail(
                &ctx.device,
                || {
                    ctx.queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        &data,
                        wgpu::TexelCopyBufferLayout {
                            offset,
                            bytes_per_row: Some(16),
                            rows_per_image: Some(1),
                        },
                        wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                    );
                },
                Some("would end up overrunning the bounds of the source buffer"),
            );
        }
    });
