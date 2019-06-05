# AutioEncoder
An experimental audio autoencoder.

# Overview
This project came out of the thought "I wonder if I could use an autoencoder to naively compress audio?".

`poc.py` contains the initial proof of concept code I used to see if a naive, simple autoencoder could even meaningfully preserve audio.
`encoder.py` contains the prototype compression routine I built after verifying some level of feasibility with my proof of concept.

# Design

The methodology used in this encoder is extremely basic (as far as machine learning can be basic).

The original wav file is chopped into small "encoding frames", with the final frame padded to length if necessary.

The samples in each of these frames are rescaled from integers in the range [-2^15, 2^15), to floats in the range [0,1), as this is easier to work with.

A five-layer bottleneck network is trained on the resulting frames (in my experiments, there were usually 50k-100k frames per song).
I settled on training for 10 epochs, using the adam optimizer and the mean square error loss function.

Once this network is trained, it is split into two halves - the encoder, which precedes the bottleneck, and the decoder, which follows it.

The encoder is then used to encode the frames, which are written to file. The decoder is also written to file, as it will be needed to decode the compressed audio.

When it is time to decode the audio, the decoder is loaded and applied to the encoded file, which lossily reproduces the float values of the frames.

These frames are then rescaled, coerced into integers, and concatenated, before being written back out as samples to a wav file.

# Usage
`python encoder.py` has two modes - Encode, and Decode.

## Encode Mode
In encode mode, the encoder is provided with a wav file to be encoded. This usage looks something like 

`python encoder.py "path/to/file.wav" "path/to/prefix"`.

If this command were invoked, `path/to/file.wav` would be encoded. This would generate two output files - `path/to/prefix.npz`, and `path/to/prefix.model`.

The `npz` file contains the actual encoded data of the wav file, and the `model` file contains the necessary model information in order to decode the `npz` file.

## Decode Mode
In decode mode, the encoder is provided with a prefix to be decoded. Following from the example above, we could invoke

`python encoder.py "path/to/prefix" "path/to/new"`.

This would load the model from `path/to/prefix.model` and use it to decode `path/to/prefix.npz`. Then, the program would rebuild a wav file and write it to `path/to/new.wav`.

# Testing

This should generally be applicable to any wav-formatted music file.

# Results

Overall, I was pretty happy with the results attained by this program.

While the resulting wav files have a large amount of noise, overall, the input music files were still highly recognizable - the song is clearly there, there is just some noise overlayed with the song. While this is certainly not ideal, it is much better results than I had expected to see.

I spent a long time tweaking hyperparameters, loss functions, optimizers, and network size, but found that none of my experiments were able to remove this noise. Thus, I believe that it is an artifact of the very simple design of my autoencoder.

# Future Work

The primary issue I would address if I continue work on this project is the issue of the noise introduced by this compression method.

Broadly speaking, I see two avenues down which I would direct such work.

## Option A - Sequence-based autoencoding

My current autoencoder network is very simple - it is not actually aware of any temporal relation between samples.

I believe that if I were to build a network based on a model that has temporal awareness, such as LSTM cells, this could potentially allow for the network to better represent the 'latent space' of the encoded audio, which would hopefully help reduce noise/loss.

One challenge with this approach is the large amount of computing resources needed to perform the training - I was unable to run an LSTM-based model on my computers. However, it is definitely a path worth considering.

## Option B - Less naivete

As mentioned, this project started to try to build a completely naive audio compressor - capable of compressing audio without any assumptions about the nature of the input data.

However, audio is a very complex domain - mapping it to a latent space is a challenging task for a network to learn.

As mentioned in the previous option, the temporal nature of the data is one such challenge - it could potentially be alleviated by using some knowledge about the nature of audio.

If I were to build a less naive network, I think I would start by processing data in windows with an FFT - this moves the data from the time domain to the frequency domain. With the temporal nature of the data removed, it would potentially be easier to model a latent space.

# Conclusion

In the end, this project was a good learning experience. I was unprepared initially for some of the challenges in representing audio data in a way understandable to a neural network, so I learned a fair amount in that regard. I also got to explore the world of autoencoding - I tried several different types of network architectures before settling on my simple deep neural network. And I was ultimately surprised at how well even such a simple model was able to create a latent space over the audio domain - despite having noise issues, the major features of the audio were preserved working only with raw samples, with no pre-built understandings of the nature of sound.
