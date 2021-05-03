#pragma once
#include "pch.h"

struct NetImpl : torch::nn::SequentialImpl
{
    NetImpl(...) {
        // Layer 1
        push_back(Conv2d(Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false)));
        push_back(MaxPool2d(2));
        push_back(Functional(torch::relu));
        // Layer 1
        push_back(Conv2d(Conv2dOptions(32, 32, 3).stride(1).padding(1).bias(false)));
        push_back(MaxPool2d(2));
        push_back(Functional(torch::relu));
        // Layer 2
        push_back(Conv2d(Conv2dOptions(32, 32, 3).stride(1).padding(1).bias(false)));
        push_back(MaxPool2d(2));
        push_back(Functional(torch::relu));
        // Layer 3
        push_back(Conv2d(Conv2dOptions(32, 32, 3).stride(1).padding(1).bias(false)));
        push_back(MaxPool2d(2));
        push_back(Functional(torch::relu));
        // Layer 3
        push_back(Conv2d(Conv2dOptions(32, 32, 3).stride(1).padding(1).bias(false)));
        push_back(MaxPool2d(2));
        push_back(Functional(torch::relu));
        // Layer 4
        push_back(Flatten());
        push_back(Linear(128, 256));
        push_back(Functional(torch::relu));
        // Layer 5
        push_back(Linear(256, 2));
        push_back(LogSoftmax(1));

    };
};
TORCH_MODULE(Net);

struct CV_DROP_BN_RELUImpl : torch::nn::SequentialImpl {
    CV_DROP_BN_RELUImpl(
        size_t _IN,
        size_t _OUT,
        size_t FILTER_SIZE,
        float DROP_RATE,
        size_t BATCHNORM
    )
    {
        push_back(Conv2d(Conv2dOptions(_IN, _OUT, FILTER_SIZE).stride(1).padding(1).bias(false)));
        if (DROP_RATE > 0 && is_training()) push_back(Dropout(DROP_RATE));
        if (BATCHNORM > 0 && is_training()) push_back(BatchNorm2d(_OUT));
        push_back(Functional(torch::relu));
    };

    torch::Tensor forward(torch::Tensor x) { return torch::nn::SequentialImpl::forward(x); }
};
TORCH_MODULE(CV_DROP_BN_RELU);

struct FEATURESImpl : torch::nn::SequentialImpl {
    FEATURESImpl(
        const vector<size_t>& CONV_LAYER,
        float  DROP_RATE = 0,
        size_t BATCHNORM = 0,
        size_t FILTER_SIZE = 3,
        size_t CHANNEL_IN = 3
    )
    {
        auto channel = CHANNEL_IN;
        for (auto N : CONV_LAYER) {
            if (N == 0)
                //push_back(Functional(AvgPool2d(AvgPool2dOptions({ 2, 2 }).stride({ 2, 2 }))));
                push_back(MaxPool2d(2));
            else
            {
                push_back(CV_DROP_BN_RELU(channel, N, FILTER_SIZE, DROP_RATE, BATCHNORM));
                channel = N;
            }
        }
    }

    torch::Tensor forward(torch::Tensor x) { return torch::nn::SequentialImpl::forward(x); }
};
TORCH_MODULE(FEATURES);

struct CLASSIFIERImpl : torch::nn::SequentialImpl {
    CLASSIFIERImpl(
        const vector<size_t>& LINEAL_LAYER,
        size_t CHANNEL_IN,
        size_t CHANNEL_OUT = 2
    )
    {
        push_back(Flatten());

        auto channel = CHANNEL_IN;
        for (auto N : LINEAL_LAYER) {
            push_back(Linear(channel, N));
            push_back(Functional(torch::relu));
            channel = N;
        }
        push_back(Linear(channel, CHANNEL_OUT));
    };

    torch::Tensor forward(torch::Tensor x) { return torch::nn::SequentialImpl::forward(x); }
};
TORCH_MODULE(CLASSIFIER);

struct VGGImpl : torch::nn::SequentialImpl {
    VGGImpl(
        size_t IMAGESIZE,
        const vector<size_t>& CONV_LAYER,
        const vector<size_t>& LINEAL_LAYER,
        float  DROP_RATE = 0,
        size_t BATCHNORM = 0,
        size_t CHANNEL_OUT = 2,
        size_t CHANNEL_IN = 3,
        size_t FILTER_SIZE = 3
    )
    {
        push_back("FEATURES", FEATURES(CONV_LAYER, DROP_RATE, BATCHNORM, FILTER_SIZE, CHANNEL_IN));
        push_back("CLASSIFIER", CLASSIFIER(LINEAL_LAYER, Size(IMAGESIZE, CONV_LAYER), CHANNEL_OUT));
        push_back(LogSoftmax(1));
    };

    size_t Size(size_t IMAGESIZE, const vector<size_t>& CONV_LAYER) const {
        // calculo cuantos elemtentos exiten en la ultima convolucion, para poder armar
        // la parte del clasificador.
        size_t count = 0;
        size_t lastfilter = 0;
        for (auto N : CONV_LAYER)
            if (N == 0) count++;
            else lastfilter = N;

        return   (IMAGESIZE >> count) * (IMAGESIZE >> count) * lastfilter;
    }

    torch::Tensor forward(torch::Tensor x) { return torch::nn::SequentialImpl::forward(x); }
};
TORCH_MODULE(VGG);