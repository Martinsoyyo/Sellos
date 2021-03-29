#include "pch.h"
#include "Trainer.h"

int main(int argc, char* argv[]) {

    Parser parser(argv[1]);

    if (parser.m_model_type == "OTRANET") {
        Net net;
        cout << net << endl;
        Trainer<Net> trainer(net, argv[1]);
    }
    else if (parser.m_model_type == "VGG") {
        VGG net(
            parser.m_image_size,
            parser.m_conv_layer_conf,
            parser.m_linear_layer_conf,
            parser.m_drop_out,
            parser.m_batch_norm,
            parser.m_output_channel,
            parser.m_input_channel
        );
        cout << net << endl;

        Trainer<VGG> trainer(net, argv[1]);

    }
    else if (parser.m_model_type == "DENSENET") {

    }

}