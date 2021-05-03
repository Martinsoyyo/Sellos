#include "pch.h"
#include "Redes.h"

#define DSEP "\\"
#define NET_ADDRESS "C:\\Users\\mmpel\\source\\repos\\Sellos\\Testeador_NN"
#define TENSOR_ADDRESS "C:\\Sellos\\Prueba_Superior"
#define TRAIN_PERCENTAGE 0.75f
#define BATCH_SIZE 16

#define IMAGE_SIZE 64
string  IMAGE_PATH = "C:\\Sellos\\Prueba_Superior\\0_sano";
//string  IMAGE_NAME = "KQHT_C3_pza=100-pos=3-img=0073-01395ms(1).jpg";
string  IMAGE_NAME = "KQHT_C3_pza=101-pos=2-img=0063-01074ms(1).jpg";
//string  IMAGE_NAME = "KQHT_C3_pza=101-pos=2-img=0067-01194ms(2).jpg";



string  IMAGE_PATH1 = "C:\\Sellos\\Prueba_Superior\\1_roto";
string  IMAGE_NAME1 = "VBME_C3_pza=002-pos=2-img=0072-01303ms(3).jpg";


template <typename NET>
size_t Test_Image(NET& m_net, string IMAGE_NAME, string IMAGE_PATH) {
    m_net->eval();

    cv::Mat src = cv::imread(IMAGE_PATH + DSEP + IMAGE_NAME);
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    cv::resize(src, src, cv::Size(IMAGE_SIZE, IMAGE_SIZE));

    auto src_tensor = torch::from_blob(src.data, { 1, 1, IMAGE_SIZE, IMAGE_SIZE }, torch::kByte);

    //cout << src_tensor.to(at::kFloat) << endl;
    //cout << src_tensor.sizes() << endl;
  
    torch::Tensor output = m_net->forward(src_tensor.to(at::kFloat).div_(255));
    torch::Tensor pred = output.argmax();

    cout << output << endl;
    cout << pred << endl;

    return(pred.item<int64_t>());
}


template <typename NET>
void Testing(NET& m_net)
{
    string IMG_ADD = TENSOR_ADDRESS + string(DSEP) + std::to_string(IMAGE_SIZE) + "x" + std::to_string(IMAGE_SIZE) + "_IMAGES.tensor";
    string TRG_ADD = TENSOR_ADDRESS + string(DSEP) + std::to_string(IMAGE_SIZE) + "x" + std::to_string(IMAGE_SIZE) + "_TARGET.tensor";

    torch::Tensor m_image;
    torch::Tensor m_target;
    torch::load(m_image, IMG_ADD);
    torch::load(m_target, TRG_ADD);

    //cout << m_image.sizes() << endl;
    //cout << m_target.sizes() << endl;


    if (m_image.sizes().size() == 3) m_image.unsqueeze_(1); 
    cout << m_target.sizes() << endl;
    cout << m_image.sizes() << endl;


    const auto N = int64_t(TRAIN_PERCENTAGE * m_image.sizes()[0]);

    torch::Tensor m_image_test = m_image.slice(0, N + 1);
    torch::Tensor m_target_test = m_target.slice(0, N + 1);

    auto IMAGE = m_image_test.split(BATCH_SIZE);
    auto TARGET = m_target_test.split(BATCH_SIZE);

    m_net->eval();

    size_t count[4] = { 0,0,0,0 };
    for (auto idx = 0; idx < IMAGE.size(); idx++) {

    //cout << IMAGE[idx] << endl;
    //cout << IMAGE[idx].sizes() << endl;

        torch::Tensor output = m_net->forward(IMAGE[idx].to(at::kFloat).div_(255));
        torch::Tensor pred = output.argmax(1);

    //    cout << "output " << endl << output << endl;
    //    cout << "pred" << endl << pred << endl;
    //    cout << "images" << endl << TARGET[idx] << endl;

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
    // VGG,B 1,D 0.000000,CL(6 6 6 0 6 6 0 6 0 6 0 ),LL(12 12 )0.999411%.pt
    vector<size_t> CL = { 8,0,8,0,8,0,8,0 };
    vector<size_t> LL = { 8,8 };
    VGG net(
        64,/*IMAGE SIZE*/
        CL,
        LL,
        0.005, /* DROP OUT*/
        1, /* BATCH NORM*/
        2, /* CHANNEL OUT*/
        1  /* CHANNEL IN*/
    );
    // Antes de cargar la RED, necesito inicializarla con la estructura que tiene,
    // todavia no se si se puede hacer solo con el archivo .pt

    torch::load(net, NET_ADDRESS + string(DSEP) + "model.pt");
    cout << net << endl;

    Testing(net);


  // cout << "sano." << endl;
  // Test_Image(net, IMAGE_NAME, IMAGE_PATH);
  // cout << endl;

   net->eval();
   auto A = torch::ones({ 1,1,64,64 });
   torch::Tensor output = net->forward(A);
   torch::Tensor pred = output.argmax(1);
   cout << output << endl;
   cout << pred << endl;
 
  // cout << "roto." << endl;
 //  Test_Image(net, IMAGE_NAME1, IMAGE_PATH1);
   
   };
