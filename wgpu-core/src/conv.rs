/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::resource;

pub fn is_power_of_two(val: u32) -> bool {
    val != 0 && (val & (val - 1)) == 0
}

pub fn is_valid_copy_src_texture_format(format: wgt::TextureFormat) -> bool {
    use wgt::TextureFormat as Tf;
    match format {
        Tf::Depth24Plus | Tf::Depth24PlusStencil8 => false,
        _ => true,
    }
}

pub fn is_valid_copy_dst_texture_format(format: wgt::TextureFormat) -> bool {
    use wgt::TextureFormat as Tf;
    match format {
        Tf::Depth32Float | Tf::Depth24Plus | Tf::Depth24PlusStencil8 => false,
        _ => true,
    }
}

pub fn map_buffer_usage(usage: wgt::BufferUsage) -> hal::BufferUse {
    let mut u = hal::BufferUse::empty();
    u.set(
        hal::BufferUse::MAP_READ,
        usage.contains(wgt::BufferUsage::MAP_READ),
    );
    u.set(
        hal::BufferUse::MAP_WRITE,
        usage.contains(wgt::BufferUsage::MAP_WRITE),
    );
    u.set(
        hal::BufferUse::COPY_SRC,
        usage.contains(wgt::BufferUsage::COPY_SRC),
    );
    u.set(
        hal::BufferUse::COPY_DST,
        usage.contains(wgt::BufferUsage::COPY_DST),
    );
    u.set(
        hal::BufferUse::INDEX,
        usage.contains(wgt::BufferUsage::INDEX),
    );
    u.set(
        hal::BufferUse::VERTEX,
        usage.contains(wgt::BufferUsage::VERTEX),
    );
    u.set(
        hal::BufferUse::UNIFORM,
        usage.contains(wgt::BufferUsage::UNIFORM),
    );
    u.set(
        hal::BufferUse::STORAGE_LOAD | hal::BufferUse::STORAGE_STORE,
        usage.contains(wgt::BufferUsage::STORAGE),
    );
    u.set(
        hal::BufferUse::INDIRECT,
        usage.contains(wgt::BufferUsage::INDIRECT),
    );
    u
}

pub fn map_texture_usage(usage: wgt::TextureUsage, aspect: hal::FormatAspect) -> hal::TextureUse {
    let mut u = hal::TextureUse::empty();
    u.set(
        hal::TextureUse::COPY_SRC,
        usage.contains(wgt::TextureUsage::COPY_SRC),
    );
    u.set(
        hal::TextureUse::COPY_DST,
        usage.contains(wgt::TextureUsage::COPY_DST),
    );
    u.set(
        hal::TextureUse::SAMPLED,
        usage.contains(wgt::TextureUsage::SAMPLED),
    );
    u.set(
        hal::TextureUse::STORAGE_LOAD | hal::TextureUse::STORAGE_STORE,
        usage.contains(wgt::TextureUsage::STORAGE),
    );
    let is_color = aspect.contains(hal::FormatAspect::COLOR);
    u.set(
        hal::TextureUse::COLOR_TARGET,
        usage.contains(wgt::TextureUsage::RENDER_ATTACHMENT) && is_color,
    );
    u.set(
        hal::TextureUse::DEPTH_STENCIL_READ | hal::TextureUse::DEPTH_STENCIL_WRITE,
        usage.contains(wgt::TextureUsage::RENDER_ATTACHMENT) && !is_color,
    );
    u
}

pub fn check_texture_dimension_size(
    dimension: wgt::TextureDimension,
    wgt::Extent3d {
        width,
        height,
        depth_or_array_layers,
    }: wgt::Extent3d,
    sample_size: u32,
    limits: &wgt::Limits,
) -> Result<(), resource::TextureDimensionError> {
    use resource::{TextureDimensionError as Tde, TextureErrorDimension as Ted};
    use std::convert::TryInto;
    use wgt::TextureDimension::*;

    let layers = depth_or_array_layers.try_into().unwrap_or(!0);
    let (extent_limits, sample_limit) = match dimension {
        D1 => (
            [
                limits.max_texture_dimension_1d,
                1,
                limits.max_texture_array_layers,
            ],
            1,
        ),
        D2 => (
            [
                limits.max_texture_dimension_2d,
                limits.max_texture_dimension_2d,
                limits.max_texture_array_layers,
            ],
            32,
        ),
        D3 => (
            [
                limits.max_texture_dimension_3d,
                limits.max_texture_dimension_3d,
                limits.max_texture_dimension_3d,
            ],
            1,
        ),
    };

    for (&dim, (&given, &limit)) in [Ted::X, Ted::Y, Ted::Z].iter().zip(
        [width, height, depth_or_array_layers]
            .iter()
            .zip(extent_limits.iter()),
    ) {
        if given == 0 {
            return Err(Tde::Zero(dim));
        }
        if given > limit {
            return Err(Tde::LimitExceeded { dim, given, limit });
        }
    }
    if sample_size == 0 || sample_size > sample_limit || !is_power_of_two(sample_size) {
        return Err(Tde::InvalidSampleCount(sample_size));
    }

    Ok(())
}

pub fn map_color_f32(color: &wgt::Color) -> [f32; 4] {
    [
        color.r as f32,
        color.g as f32,
        color.b as f32,
        color.a as f32,
    ]
}
pub fn map_color_i32(color: &wgt::Color) -> [i32; 4] {
    [
        color.r as i32,
        color.g as i32,
        color.b as i32,
        color.a as i32,
    ]
}
pub fn map_color_u32(color: &wgt::Color) -> [u32; 4] {
    [
        color.r as u32,
        color.g as u32,
        color.b as u32,
        color.a as u32,
    ]
}

/// Take `value` and round it up to the nearest alignment `alignment`.
///
/// ```text
/// (0, 3) -> 0
/// (1, 3) -> 3
/// (2, 3) -> 3
/// (3, 3) -> 3
/// (4, 3) -> 6
/// ...
pub fn align_up(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
}
