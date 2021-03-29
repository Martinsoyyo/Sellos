#include "pch.h"
#include "Redes.h"
using namespace std;

int main(int argc, char* argv[]) {

   vector<size_t> CL = { 8,0,8,0,8,0,8,0,8,0 };
    vector<size_t> LL = { 8 };

    VGG net(
        64,
        CL,
        LL,
        0.2,
        1,
        2,
        1
    );
		
	torch::Tensor A = torch::randn({ 1,1,64,64 });

    

    cout << A.sizes() << endl;
    cout << net << endl;
    auto output = net->forward(A);
    std::cout << output << endl;

    torch::load(net, "C:\\Users\\mmpel\\source\\repos\\Sellos\\Testeador_NN\\model.pt");
    output = net->forward(A);
    std::cout << output << endl;


};
