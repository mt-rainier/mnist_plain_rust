# MNIST in Plain Rust

This is a single-layer MNIST neural network implemented in plain Rust.
It's not optimized. It's my first Rust program.

Running `cargo run` will start training and prints info about accuracy and loss.

I'll will optimize the performance after I learn more about Rust.

## Performance

Release version takes 1m38s for 1000 iterations, on 8-core machine with 32 GB
mem.

The C version referenced below takes 3m56s for the same setting.

## Credit

Current version is translated from
https://github.com/AndrewCarterUK/mnist-neural-network-plain-c.  Thanks, Andrew!

Also thanks to the Rust community, specifically this [Discord
channel](https://discord.com/invite/aVESxV8) for answering my questions.
