use bevy::{prelude::*, render::{texture::ImageSampler, render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages}}, asset::HandleId, window::PrimaryWindow};

use candle_core::{Device, Result, DType, Tensor, D};
use candle_nn::{Linear, Module, VarMap, VarBuilder};

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;
const WINDOW_SIZE: Vec2 = Vec2::new(720.0, 720.0);

#[derive(Resource)]
pub struct TrainedLinear {
    model: LinearModel,
}

#[derive(Resource)]
pub struct ImageId {
    id: HandleId,
}

#[derive(Resource)]
pub struct MyDevice {
    device: Device,
}

fn linear_z(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Linear> {
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::ZERO)?;
    let bs = vs.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(ws, Some(bs)))
}

struct LinearModel {
    linear: Linear,
}

impl LinearModel {
    fn new(vs: VarBuilder) -> Result<Self> {
        let linear = linear_z(IMAGE_DIM, LABELS, vs)?;
        Ok(Self { linear })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }
}


fn main() -> Result<()> {
    
    let dev = Device::cuda_if_available(0)?;

    let mut varmap = VarMap::new();    
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = LinearModel::new(vs.clone())?;
    varmap.load("linear.safetensors")?;


    App::new()
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin {
                    default_sampler: ImageSampler::nearest_descriptor(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "drawing".into(),
                        resolution: WINDOW_SIZE.into(),
                        resizable: false,
                        ..default()
                    }),
                    ..Default::default()
                })
        )
        .insert_resource(MyDevice{ device: dev })
        .insert_resource(ClearColor(Color::RED))
        .add_systems(Startup, startup)
        .add_systems(Update, (click_system, guess_system))
        .insert_resource(TrainedLinear { model })
        .run();

    Ok(())

}

fn startup (
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {

    commands
        .spawn(Camera2dBundle::default());

    let data = vec![255; 4*28*28];

    let mut image = Image::new(
        Extent3d {
            width: 28,
            height: 28,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let image = images.add(image);
    let id = image.id();
    
    commands.insert_resource(ImageId { id });

    commands.spawn(SpriteBundle {
        sprite: Sprite {
            custom_size: Some(WINDOW_SIZE),
            ..default()
        },
        
        texture: image,
        ..default()
    });

}

fn click_system(
    buttons: Res<Input<MouseButton>>,
    id: Res<ImageId>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    if buttons.pressed(MouseButton::Left) {
        let handle = Handle::weak(id.id);

        if let Some(image) = images.get_mut(&handle) {

            if let Some(position) = windows.single().cursor_position() {
                let position = (
                    (position.x * 28.0 / WINDOW_SIZE.x) as usize,
                    (position.y * 28.0 / WINDOW_SIZE.y) as usize,
                );

                let index = position.0 + (position.1 * 28);

                image.data[index * 4] = 0;
                image.data[(index * 4) + 1] = 0;
                image.data[(index * 4) + 2] = 0;
                image.data[(index * 4) + 3] = 255;
            }
        }
    }
}

fn guess_system(
    id: Res<ImageId>,
    mut images: ResMut<Assets<Image>>,
    device: Res<MyDevice>,
    model: Res<TrainedLinear>,
) {

    let handle = Handle::weak(id.id);
    let mut vec: Vec<f32> = Vec::new();

    if let Some(image) = images.get_mut(&handle) {

        for (i, a) in image.data.iter().enumerate() {

            if i % 4 == 0 {

                if *a == 0 {
                    vec.push(1.0)
                } else {
                    vec.push(0.0)
                }

            }

        }

        let tensor: Tensor = Tensor::from_vec(vec, (1, 784), &device.device).unwrap();

        let a = model.model.forward(&tensor).unwrap();


        let a = a
            .argmax(D::Minus1).unwrap().get(0).unwrap().to_scalar::<u32>().unwrap();

        println!("{a}");

    }

}