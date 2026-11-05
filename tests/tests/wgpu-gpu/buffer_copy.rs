//! Tests for buffer copy validation.

use wgpu::BufferAddress;

use wgpu_test::{
    apply, fail_if, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(COPY_ALIGNMENT);
    vec.push(QUEUE_WRITE_BUFFER_VIEW);
}

#[apply(gpu_test!)]
static QUEUE_WRITE_BUFFER_VIEW: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("queue write view destination"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer_with(&buffer, 0, 16.try_into().expect("nonempty upload"))
            .expect("valid upload view")
            .copy_from_slice(&[0x5a; 16]);
        ctx.queue.submit([]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |result| {
            result.expect("uploaded buffer mapping");
        });
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .expect("upload completion");
        assert_eq!(
            &*buffer.slice(..).get_mapped_range().expect("uploaded bytes"),
            &[0x5a; 16]
        );
        buffer.unmap();
    });

fn try_copy(
    ctx: &wgpu_test::TestingContext,
    offset: BufferAddress,
    size: BufferAddress,
    error_message: Option<&'static str>,
) {
    let buffer = ctx.device.create_buffer(&BUFFER_DESCRIPTOR);
    let data = vec![255; size as usize];

    fail_if(
        &ctx.device,
        error_message.is_some(),
        || ctx.queue.write_buffer(&buffer, offset, &data),
        error_message,
    );
}

#[apply(gpu_test!)]
static COPY_ALIGNMENT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        try_copy(&ctx, 0, 0, None);
        try_copy(
            &ctx,
            4,
            16 + 1,
            Some("copy size 17 does not respect `copy_buffer_alignment`"),
        );
        try_copy(
            &ctx,
            64,
            20 + 2,
            Some("copy size 22 does not respect `copy_buffer_alignment`"),
        );
        try_copy(
            &ctx,
            256,
            44 + 3,
            Some("copy size 47 does not respect `copy_buffer_alignment`"),
        );
        try_copy(&ctx, 1024, 8 + 4, None);

        try_copy(&ctx, 0, 4, None);
        try_copy(
            &ctx,
            4 + 1,
            8,
            Some("buffer offset 5 is not aligned to block size or `copy_buffer_alignment`"),
        );
        try_copy(
            &ctx,
            64 + 2,
            12,
            Some("buffer offset 66 is not aligned to block size or `copy_buffer_alignment`"),
        );
        try_copy(
            &ctx,
            256 + 3,
            16,
            Some("buffer offset 259 is not aligned to block size or `copy_buffer_alignment`"),
        );
        try_copy(&ctx, 1024 + 4, 4, None);
    });

const BUFFER_SIZE: BufferAddress = 1234;

const BUFFER_DESCRIPTOR: wgpu::BufferDescriptor = wgpu::BufferDescriptor {
    label: None,
    size: BUFFER_SIZE,
    usage: wgpu::BufferUsages::COPY_SRC.union(wgpu::BufferUsages::COPY_DST),
    mapped_at_creation: false,
};
