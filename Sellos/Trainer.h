#pragma once
#include "pch.h"
#include "Parser.h"
#include "Redes.h"

template<class NET>
class Trainer {
public:
    Trainer(NET& OBJ, const string& NET_CONFIG);

    void Load_Tensor_From_File();
    void Move_Tensor_To_GPU();
    void Separate_Tensor_For_Train_And_Test();
    void Save_Model_To_CPU(const float&);
    void Confusion_Matrix();
    void Run();

    void Train(size_t epoch, torch::optim::Optimizer& optimizer);
    float Test(const torch::Tensor& IMAGE, const torch::Tensor& TARGET);

private:
    torch::Device m_device;
    // Tenosres donde se encuentran los datos aprocesar.
    torch::Tensor m_image;
    torch::Tensor m_target;

    // Referencias que apuntan a la zona de testeo y entrenamiento.
    torch::Tensor m_image_train;
    torch::Tensor m_target_train;
    torch::Tensor m_image_test;
    torch::Tensor m_target_test;

    Parser m_parser;
    NET m_net;
};

template<class NET>
Trainer<NET>::Trainer(NET& OBJ, const string& NET_CONFIG) : m_device(torch::kCPU), m_net(OBJ), m_parser(NET_CONFIG) { Run(); }

template<class NET>
void Trainer<NET>::Load_Tensor_From_File() {
    try {
        string IMG_ADD = m_parser.m_root_folder + "//" + std::to_string(m_parser.m_image_size) + "x" + std::to_string(m_parser.m_image_size) + "_IMAGES.tensor";
        string TRG_ADD = m_parser.m_root_folder + "//" + std::to_string(m_parser.m_image_size) + "x" + std::to_string(m_parser.m_image_size) + "_TARGET.tensor";

        cout << "Cargando " << IMG_ADD << endl;
        cout << "Cargando " << TRG_ADD << endl;
        torch::load(m_image, IMG_ADD);
        torch::load(m_target, TRG_ADD);

        if (m_image.sizes().size() == 3) m_image.unsqueeze_(1);
    }
    catch (exception& e) {
        cerr << "Trainer::Trainer() - torch::load" << endl << e.what();
        throw(e);
    }
};

template<class NET>
void Trainer<NET>::Separate_Tensor_For_Train_And_Test() {
    const auto N = int64_t(m_parser.m_percent_to_train * m_image.size(0));

    m_image_train = m_image.slice(0, 0, N);
    m_image_test = m_image.slice(0, N + 1);
    m_target_train = m_target.slice(0, 0, N);
    m_target_test = m_target.slice(0, N + 1);

};

template<class NET>
void Trainer<NET>::Move_Tensor_To_GPU() {
    if (torch::cuda::is_available()) {
        cout << "CUDA available! Training on GPU." << endl;
        m_device = torch::kCUDA;
    }
    else {
        cout << "Training on CPU." << endl;
        m_device = torch::kCPU;
    }

    m_net->to(m_device);
    m_image.to(m_device);
    m_target.to(m_device);
};

template<class NET>
void Trainer<NET>::Save_Model_To_CPU(const float& RES) {
    torch::load(m_net, "model.pt");
    m_net->to(torch::kCPU, true);

    string str =
        m_parser.m_model_type + "," +
        "B " + (m_parser.m_batch_norm ? "1" : "0") + "," +
        "D " + to_string(m_parser.m_drop_out) + ",";

    str += "CL(";
    for (auto N : m_parser.m_conv_layer_conf) str += to_string(N) + " ";
    str += "),LL(";
    for (auto N : m_parser.m_linear_layer_conf) str += to_string(N) + " ";
    str += ")" + to_string(RES) + "%.pt";

    torch::save(m_net, str);
}

template<class NET>
float Trainer<NET>::Test(const torch::Tensor& __IMAGE, const torch::Tensor& __TARGET)
{
    auto IMAGE = __IMAGE.split(m_parser.m_batch_size);
    auto TARGET = __TARGET.split(m_parser.m_batch_size);

    torch::NoGradGuard no_grad;
    m_net->eval();

    int32_t correct = 0;
    for (auto idx = 0; idx < IMAGE.size(); idx++) {

        auto output = m_net->forward(IMAGE[idx].to(m_device).to(at::kFloat).div_(255));
        auto pred = output.argmax(1);

        correct += pred.eq(TARGET[idx].to(m_device).to(at::kLong)).sum().template item<int64_t>();
    }

    return (static_cast<float>(correct) / __IMAGE.sizes()[0]);
}

template<class NET>
void Trainer<NET>::Train(size_t epoch, torch::optim::Optimizer& optimizer)
{
    m_net->train();
    auto IMAGE = m_image_train.split(m_parser.m_batch_size);
    auto TARGET = m_target_train.split(m_parser.m_batch_size);

    //torch::Tensor weigth = torch::zeros({ 1,2 }).to(m_device);
    //weigth[0][0] = 1;
    //weigth[0][1] = 2;

    for (auto idx = 0; idx < IMAGE.size(); idx++) {
        optimizer.zero_grad();
        auto output = m_net->forward(IMAGE[idx].to(m_device).to(at::kFloat).div_(255));

        auto loss = torch::nll_loss(output, TARGET[idx].to(m_device).to(at::kLong)/*,weigth*/);
        loss.backward();
        optimizer.step();

        std::printf("\r Epoch: %ld [%5ld/%5ld] Loss: %.4f |", epoch, idx * m_parser.m_batch_size, m_image_train.sizes()[0], loss.template item<float>());
    }
}

template<class NET>
void Trainer<NET>::Confusion_Matrix()
{
    torch::load(m_net, "model.pt");
    m_net->to(m_device, true);

    auto IMAGE = m_image_test.to(m_device).split(m_parser.m_batch_size);
    auto TARGET = m_target_test.to(m_device).split(m_parser.m_batch_size);

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


template<class NET>
void Trainer<NET>::Run() {
    srand((unsigned)time(0));
    torch::manual_seed(rand() % RAND_MAX);

    Load_Tensor_From_File();
    Separate_Tensor_For_Train_And_Test();
    Move_Tensor_To_GPU();

    torch::optim::SGD optimizer(m_net->parameters(), torch::optim::SGDOptions(0.1).momentum(0.5));
    //torch::optim::Adam optimizer(m_net->parameters(), torch::optim::AdamOptions(0.0001).betas(std::make_tuple(0.9, 0.995)).eps(1e-8).weight_decay(0));


    auto best_result = Test(m_image_test, m_target_test);

    for (size_t epoch = 1; epoch <= m_parser.m_epoch; ++epoch) {
        Train(epoch, optimizer);

        float result = Test(m_image_test, m_target_test);
        std::printf("Train set: Accuracy: %.3f  ,  Test set: Accuracy: %.3f \n", Test(m_image_train, m_target_train), result);

        if (result > best_result) {
            torch::save(m_net, "model.pt");
            best_result = result;
        }
    }

    std::printf("Best Accuracy: %.3f \n", best_result);
    Confusion_Matrix();
    Save_Model_To_CPU(best_result); // Trae de disco la mejor configuracion encontrada y la vuelve a grabar con la configuracion en el nombre de la red.
};