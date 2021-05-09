#pragma once
#include "pch.h"
#include "Redes.h"

using namespace torch::nn;

struct DenseLayerImpl : SequentialImpl {
    DenseLayerImpl(
        int64_t _IN,  //Num de canales de imagen
        int64_t _1x1, //Num de filtros de 1x1
        int64_t _3x3, //Num de filtros de 3x3
        float DROP_RATE
    )
    {
        push_back(Conv2d(Conv2dOptions(_IN, _1x1, 1).stride(1).bias(false)));
        push_back(BatchNorm2d(_1x1));
        push_back(Functional(torch::relu));
        push_back(Conv2d(Conv2dOptions(_1x1, _3x3, 3).stride(1).padding(1).bias(false)));

        if (DROP_RATE > 0 && is_training()) push_back(Dropout(DROP_RATE));
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::cat({ x, SequentialImpl::forward(x) }, 1);
    }
};
TORCH_MODULE(DenseLayer);

struct TransitionImpl : torch::nn::SequentialImpl {
    TransitionImpl(
        int64_t _IN,  //Num de canales de imagen
        int64_t _1x1  //Num de filtros de 1x1
    )
    {
        push_back(Conv2d(Conv2dOptions(_IN, _1x1, 1).stride(1).bias(false)));
        push_back(BatchNorm2d(_1x1));
        push_back(Functional(torch::relu));
        //push_back(Functional(AvgPool2d(AvgPool2dOptions({ 2, 2 }).stride({ 2, 2 }))));
        push_back(MaxPool2d(2));
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::nn::SequentialImpl::forward(x);
    }
};
TORCH_MODULE(Transition);

struct DenseNetImpl : torch::nn::SequentialImpl {
    DenseNetImpl(
        size_t IMAGESIZE,
        const vector<string>& CONV_LAYER,
        const vector<string>& LINEAL_LAYER,
        size_t INPUT_CHANNEL,
        size_t OUT_CHANNEL
    )
    {
        //INIT
        auto channel = stoi(CONV_LAYER[0]);
        auto imagesize = IMAGESIZE;
        push_back(Conv2d(Conv2dOptions(INPUT_CHANNEL, channel, 3).stride(1).padding(1).bias(false)));

        //FEATURES
        for (auto N = 1; N < CONV_LAYER.size(); N++) {
            if (CONV_LAYER[N] == "D") //Dense Layer
            {
                volatile auto _1x1 = stoi(CONV_LAYER[++N]);   //Num de filtros de 1x1
                volatile auto _3x3 = stoi(CONV_LAYER[++N]);   //Num de filtros de 3x3
                volatile auto DropOut = stof(CONV_LAYER[++N]);  //DropOut
                push_back(DenseLayer(channel, _1x1, _3x3, DropOut));

                channel += _3x3;
            }
            else if (CONV_LAYER[N] == "T") //Transition Layer;
            {
                volatile auto _1x1 = stoi(CONV_LAYER[++N]);   //Num de filtros de 1x1
                push_back(Transition(channel, _1x1));

                channel = _1x1;
                imagesize *= 0.5;
            }
            else break;
        }

        //CLASSIFIER
        push_back(Flatten());
        volatile auto lineal_layer = channel * imagesize * imagesize;
        for (auto N = 0; N < LINEAL_LAYER.size(); N++) {
            try {
                volatile auto NUM = stoi(LINEAL_LAYER[N]);
                push_back(Linear(lineal_layer, NUM));
                push_back(Functional(torch::relu));
                lineal_layer = NUM;
            }
            catch (...) { break; }
        };

        push_back(Linear(lineal_layer, OUT_CHANNEL));
        push_back(LogSoftmax(1));
    };
};
TORCH_MODULE(DenseNet);