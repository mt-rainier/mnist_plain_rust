use rand::Rng;
use std::error::Error;
use std::fs::File;
use std::io::SeekFrom;
use std::io::prelude::*;
use std::io;
use std::mem;
use byteorder::{LittleEndian, NativeEndian};
use std::any::TypeId;
use std::convert::TryInto;

const MNIST_LABEL_MAGIC: u32 = 0x00000801;
const MNIST_IMAGE_MAGIC: u32 = 0x00000803;
const MNIST_IMAGE_WIDTH: u32 = 28;
const MNIST_IMAGE_HEIGHT: u32 = 28;
const MNIST_IMAGE_SIZE: usize = MNIST_IMAGE_WIDTH as usize * MNIST_IMAGE_HEIGHT as usize;
const MNIST_LABELS: usize = 10;

const STEPS: usize = 1000;
const BATCH_SIZE: usize = 100;

// Downloaded from: http://yann.lecun.com/exdb/mnist/
const TRAIN_IMAGES_FILE: &str = "data/train-images-idx3-ubyte";
const TRAIN_LABELS_FILE: &str = "data/train-labels-idx1-ubyte";
const TEST_IMAGES_FILE: &str = "data/t10k-images-idx3-ubyte";
const TEST_LABELS_FILE: &str = "data/t10k-labels-idx1-ubyte";

struct MnistImage {
    pixels: [u8; MNIST_IMAGE_SIZE],
}

impl MnistImage {
    fn new(pixels: [u8; MNIST_IMAGE_SIZE]) -> Self {
        Self {
            pixels
        }
    }
}

struct MnistDataset {
    images: Vec<MnistImage>,
    labels: Vec<u8>,
    size: usize,
}

struct MnistDatasetSlice<'a> {
    images: &'a[MnistImage],
    labels: &'a[u8],
    size: usize,
}

struct NeuralNetwork {
    b: [f32; MNIST_LABELS],
    w: [[f32; MNIST_IMAGE_SIZE]; MNIST_LABELS],
}

struct MnistImageFileHeader {
    magic_number: u32,
    number_of_images: u32,
    number_of_rows: u32,
    number_of_columns: u32,
}

struct MnistLabelFileHeader {
    magic_number: u32,
    number_of_labels: u32,
}

struct NeuralNetworkGradient {
    b_grad: [f32; MNIST_LABELS],
    w_grad: [[f32; MNIST_IMAGE_SIZE]; MNIST_LABELS],
}

fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn get_labels(path: &str) -> io::Result<Vec<u8>> {
    let mut f = File::open(path)?;

    let header = MnistLabelFileHeader {
        magic_number: read_u32(&mut f)?,
        number_of_labels: read_u32(&mut f)?,
    };

    if MNIST_LABEL_MAGIC != header.magic_number {
        return Err(io::Error::new(io::ErrorKind::Other, "magic number doesn't match!"));
    }

    let mut labels: Vec<u8> = Vec::with_capacity(header.number_of_labels as usize);

    f.read_to_end(&mut labels)?;

    return Ok(labels);
}

fn get_images(path: &str) -> io::Result<Vec<MnistImage>> {
    let mut f = File::open(path)?;

    let header = MnistImageFileHeader {
        magic_number: read_u32(&mut f)?,
        number_of_images: read_u32(&mut f)?,
        number_of_rows: read_u32(&mut f)?,
        number_of_columns: read_u32(&mut f)?,
    };

    if MNIST_IMAGE_MAGIC != header.magic_number {
        return Err(io::Error::new(io::ErrorKind::Other, "magic number doesn't match!"));
    }

    if MNIST_IMAGE_WIDTH != header.number_of_rows {
        return Err(io::Error::new(io::ErrorKind::Other, "row number doesn't match!"));
    }

    if MNIST_IMAGE_HEIGHT != header.number_of_columns {
        return Err(io::Error::new(io::ErrorKind::Other, "column number doesn't match!"));
    }

    let mut images: Vec<MnistImage> = Vec::with_capacity(header.number_of_images as usize);

    for i in 0..header.number_of_images {
        let mut buf = [0u8; mem::size_of::<MnistImage>()];
        f.read_exact(&mut buf[..])?;
        // when pushed to Vec, it's moved from stack to heap
        images.push(MnistImage::new(buf));
    }

    return Ok(images);
}

fn mnist_get_dataset(images_file: &str, labels_file: &str) -> Result<MnistDataset, io::Error> {
    let mut dataset = MnistDataset {
        images: match get_images(images_file) {
            Err(e) => return Err(e),
            Ok(imgs) => imgs,
        },
        labels: match get_labels(labels_file) {
            Err(e) => return Err(e),
            Ok(labels) => labels,
        },
        size: 0,
    };

    assert!(dataset.images.len() == dataset.labels.len());

    dataset.size = dataset.images.len();

    Ok(dataset)
}

fn init_nn() -> NeuralNetwork {
    let mut rng = rand::thread_rng();
    let mut network = NeuralNetwork {
        b: [rng.gen(); MNIST_LABELS],
        w: [[rng.gen(); MNIST_IMAGE_SIZE]; MNIST_LABELS]
    };
    network
}

fn mnist_batch(dataset: &MnistDataset, mut size: usize, index: usize) -> Result<MnistDatasetSlice, &str> {
    let start_offset = size * index;

    if start_offset >= dataset.size {
        return Err("Beyond dataset size");
    }

    if start_offset + size > dataset.size {
        size = dataset.size - start_offset;
    }

    let mut batch = MnistDatasetSlice {
        images: &dataset.images[start_offset..(start_offset + size)],
        labels: &dataset.labels[start_offset..(start_offset + size)],
        size: size
    };

    return Ok(batch);
}

fn pixel_scale(p: u8) -> f32 {
    p as f32 / 255.0
}

// Calculate the softmax vector from the activations. This uses a more
// numerically stable algorithm that normalises the activations to prevent
// large exponents.
fn neural_network_softmax(activations: &mut [f32; MNIST_LABELS]) {
    let mut max = activations[0];
    for i in 0..activations.len() {
        if activations[i] > max {
            max = activations[i];
        }
    }

    let mut sum = 0.0;
    for i in 0..activations.len() {
        activations[i] = (activations[i] - max).exp();
        sum += activations[i];
    }

    for i in 0..activations.len() {
        activations[i] /= sum;
    }
}

// Use the weights and bias vector to forward propogate through the neural
// network and calculate the activations.
fn neural_network_hypothesis(image: &MnistImage, network: &NeuralNetwork, activations: &mut [f32;
                             MNIST_LABELS]) {
    for i in 0..MNIST_LABELS {
        activations[i] = network.b[i];

        for j in 0..MNIST_IMAGE_SIZE {
            activations[i] += network.w[i][j] * pixel_scale(image.pixels[j]);
        }
    }

    neural_network_softmax(activations);
}

// Update the gradients for this step of gradient descent using the gradient
// contributions from a single training example (image).
//
// This function returns the loss ontribution from this training example.
fn neural_network_gradient_update(image: &MnistImage, network: &NeuralNetwork, gradient:
                                  &mut NeuralNetworkGradient, label: u8) -> f32 {
    let mut activations = [0.0; MNIST_LABELS];
    //float b_grad, W_grad;
    let mut b_grad = 0.0;
    let mut w_grad = 0.0;

    // First forward propagate through the network to calculate activations
    neural_network_hypothesis(&image, &network, &mut activations);

    for i in 0..MNIST_LABELS {
        // This is the gradient for a softmax bias input
        b_grad = if i == label as usize { activations[i] - 1.0 } else { activations[i] };

        for j in 0..MNIST_IMAGE_SIZE {
            // The gradient for the neuron weight is the bias multiplied by the input weight
            w_grad = b_grad * pixel_scale(image.pixels[j]);

            // Update the weight gradient
            gradient.w_grad[i][j] += w_grad;
        }

        // Update the bias gradient
        gradient.b_grad[i] += b_grad;
    }

    //// Cross entropy loss
    0.0 - (activations[label as usize]).ln()
}


// Run one step of gradient descent and update the neural network.
fn neural_network_training_step(dataset: &MnistDatasetSlice, network: &mut NeuralNetwork,
                                learning_rate: f32) -> f32 {
    let mut gradient = NeuralNetworkGradient {
        b_grad: [0.0; MNIST_LABELS],
        w_grad: [[0.0; MNIST_IMAGE_SIZE]; MNIST_LABELS]
    };
    let mut total_loss = 0.0;

    // Calculate the gradient and the loss by looping through the training set
    for i in 0..dataset.size {
        total_loss += neural_network_gradient_update(&dataset.images[i], &network, &mut gradient, dataset.labels[i]);
    }

    // Apply gradient descent to the network
    for i in 0..MNIST_LABELS {
        network.b[i] -= learning_rate * gradient.b_grad[i] / dataset.size as f32;

        for j in 0..MNIST_IMAGE_SIZE {
            network.w[i][j] -= learning_rate * gradient.w_grad[i][j] / dataset.size as f32;
        }
    }

    total_loss
}

// Calculate the accuracy of the predictions of a neural network on a dataset.
fn calculate_accuracy(dataset: &MnistDataset, network: &NeuralNetwork) -> f32 {
    let mut activations = [0.0; MNIST_LABELS];
    let mut correct = 0;
    let mut predict = 0u8;

    // Loop through the dataset
    for i in 0..dataset.size {
        // Calculate the activations for each image using the neural network
        neural_network_hypothesis(&dataset.images[i], &network, &mut activations);

        // Set predict to the index of the greatest activation
        let mut max_activation = activations[0];
        predict = 0;
        for j in 0..MNIST_LABELS {
            if max_activation < activations[j] {
                max_activation = activations[j];
                predict = j as u8;
            }
        }

        // Increment the correct count if we predicted the right label
        if predict == dataset.labels[i] {
            correct += 1;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    correct as f32 / dataset.size as f32
}

fn main() {
    let mut loss: f32;
    let mut accuracy: f32;
    let i: usize;
    let batches: usize;

    // Read the datasets from the files
    let train_dataset = mnist_get_dataset(TRAIN_IMAGES_FILE, TRAIN_LABELS_FILE)
        .expect("Failed to get training dataset");
    let test_dataset = mnist_get_dataset(TEST_IMAGES_FILE, TEST_LABELS_FILE)
    .expect("Failed to get test dataset");

    // Initialize weights and biases with random values
    let mut network: NeuralNetwork = init_nn();

    // Calculate how many batches (so we know when to wrap around)
    batches = train_dataset.size / BATCH_SIZE;
    assert!(batches != 0);

    for i in 0..STEPS {
        // Initialize a new batch
        let batch = mnist_batch(&train_dataset, 100, i % batches).expect("Failed to slice a batch");

        // Run one step of gradient descent and calculate the loss
        loss = neural_network_training_step(&batch, &mut network, 0.5);

        // Calculate the accuracy using the whole test dataset
        accuracy = calculate_accuracy(&test_dataset, &network);

        println!("Step {}\tAverage Loss: {}\tAccuracy: {}\n", i, loss / batch.size as f32, accuracy);
    }

    println!("Hello, world!");
}
