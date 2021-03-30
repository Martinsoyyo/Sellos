#pragma once
#include "pch.h"


template<class NET>
class Tester {
public:
    Tester(NET net, const string& ROOT, const size_t& IMAGE_SIZE, const float& PERCENT_TO_TRAIN);
    void Load_Tensor_From_File();
    torch::Tensor Confusion_Matrix();
    void Separate_Tensor_For_Train_And_Test();
    void Test();

private:
    string m_root_folder;
    size_t m_image_size;
    float m_percent_to_train;

    torch::Tensor m_image;
    torch::Tensor m_target;
    NET m_net;

    // Referencias que apuntan a la zona de testeo y entrenamiento.
    torch::Tensor m_image_train;
    torch::Tensor m_target_train;
    torch::Tensor m_image_test;
    torch::Tensor m_target_test;

};

template<class NET>
Tester<NET>::Tester(NET net, const string& ROOT, const size_t& IMAGE_SIZE, const float& PERCENT_TO_TRAIN) :
    m_net(net), m_image_size(IMAGE_SIZE), m_root_folder(ROOT), m_percent_to_train(PERCENT_TO_TRAIN)
{
    Load_Tensor_From_File();
    Separate_Tensor_For_Train_And_Test();
    Test();
};

template<class NET>
void Tester<NET>::Load_Tensor_From_File() {
    try {
        string IMG_ADD = m_root_folder + "//" + to_string(m_image_size) + "x" + to_string(m_image_size) + "_IMAGES.tensor";
        string TRG_ADD = m_root_folder + "//" + to_string(m_image_size) + "x" + to_string(m_image_size) + "_TARGET.tensor";
        
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
void Tester<NET>::Test()
{
    size_t BATCHSIZE = 128;
    auto IMAGE = m_image_test.split(BATCHSIZE);
    auto TARGET = m_target_test.split(BATCHSIZE);

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
void Tester<NET>::Separate_Tensor_For_Train_And_Test() {
    const auto N = int64_t(m_percent_to_train * m_image.size(0));

    m_image_train = m_image.slice(0, 0, N);
    m_image_test = m_image.slice(0, N + 1);
    m_target_train = m_target.slice(0, 0, N);
    m_target_test = m_target.slice(0, N + 1);

};
