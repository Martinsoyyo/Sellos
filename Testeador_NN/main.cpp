#include "pch.h"
#include "Redes.h"
#include "Tester.h"

#define DSEP "\\"
#define NET_ADDRESS "C:\\Users\\mmpel\\source\\repos\\Sellos\\Testeador_NN"
#define TENSOR_ADDRESS "C:\\Sellos\\Prueba_Superior"
#define TRAIN_PERCENTAGE 0.75f
#define IMAGE_SIZE 64
#define BATCH_SIZE 128


template <typename NET>
void Testing(NET& m_net)
{
    string IMG_ADD = TENSOR_ADDRESS + string(DSEP) + std::to_string(IMAGE_SIZE) + "x" + std::to_string(IMAGE_SIZE) + "_IMAGES.tensor";
    string TRG_ADD = TENSOR_ADDRESS + string(DSEP) + std::to_string(IMAGE_SIZE) + "x" + std::to_string(IMAGE_SIZE) + "_TARGET.tensor";

    torch::Tensor m_image;
    torch::Tensor m_target;
    torch::load(m_image, IMG_ADD);
    torch::load(m_target, TRG_ADD);
    if (m_image.sizes().size() == 3) m_image.unsqueeze_(1);

    const auto N = int64_t(TRAIN_PERCENTAGE * m_image.sizes()[0]);

    torch::Tensor m_image_test = m_image.slice(0, N + 1);
    torch::Tensor m_target_test = m_target.slice(0, N + 1);

    auto IMAGE = m_image_test.split(BATCH_SIZE);
    auto TARGET = m_target_test.split(BATCH_SIZE);

    torch::NoGradGuard no_grad;
    m_net->eval();

    size_t count[4] = { 0,0,0,0 };
    for (auto idx = 0; idx < IMAGE.size(); idx++) {

        torch::Tensor output = m_net->forward(IMAGE[idx].to(at::kFloat).div_(255));
        torch::Tensor pred = output.argmax(1);

        torch::Tensor OK_1_1 = pred.logical_and(TARGET[idx]);
        torch::Tensor OK_0_0 = pred.logical_or(TARGET[idx]).logical_not();
        torch::Tensor FALSO_POSITIVO = pred.logical_and(TARGET[idx].logical_not());
        torch::Tensor FALSO_NEGATIVO = pred.logical_not().logical_and(TARGET[idx]);

        count[0] += OK_1_1.sum().item<int64_t>();
        count[1] += OK_0_0.sum().item<int64_t>();
        count[2] += FALSO_POSITIVO.sum().item<int64_t>();
        count[3] += FALSO_NEGATIVO.sum().item<int64_t>();

    }

    cout << "Testeados Totales: " << m_image_test.sizes()[0] << endl;
    cout << "OK_1_1: " << count[0] << endl;
    cout << "OK_0_0: " << count[1] << endl;
    cout << "FALSO_POSITIVO: " << count[2] << endl;
    cout << "FALSO_NEGATIVO: " << count[3] << endl;
}


int main(int argc, char* argv[]) {
    // Tomando de ejemplo que el modelo se llama como a continuacion, la definicion de la red es asi..
    // VGG,B 1,D 0.000000,CL(8 8 8 0 8 8 0 8 0 8 0 8 0 ),LL(32 )0.997644%
    vector<size_t> CL = { 8,4,8,0,8,8,0,8,0,8,0,8,0 };
    vector<size_t> LL = { 32 };
    VGG net(
        64,/*IMAGE SIZE*/
        CL,
        LL,
        0, /* DROP OUT*/
        1, /* BATCH NORM*/
        2, /* CHANNEL OUT*/
        1  /* CHANNEL IN*/
    );
    // Antes de cargar la RED, necesito inicializarla con la estructura que tiene,
    // todavia no se si se puede hacer solo con el archivo .pt

    torch::load(net, NET_ADDRESS + string(DSEP) + "model.pt");
    cout << net;

    Testing(net);
};
