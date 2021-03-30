#include "pch.h"
#include "Redes.h"
#include "Tester.h"
#define DSEP "\\"


int main(int argc, char* argv[]) {
    // Tomando de ejemplo que el modelo se llama como a continuacion, la definicion de la red es asi..
    //VGG,B 1,D 0.000000,CL(8 8 8 0 8 8 0 8 0 8 0 8 0 ),LL(32 )0.997646%
    vector<size_t> CL = { 8,8,8,0,8,8,0,8,0,8,0,8,0 };
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
    torch::load(net, "C:\\Users\\mmpel\\source\\repos\\Sellos\\Testeador_NN\\model.pt");

    // Tester C:\\Sellos\\Prueba_Superior 64 0.75
    Tester<VGG> Tester(
        net,
        argv[1],              // Direccion de los Tensores
        stoi(string(argv[2])),// Tamano de la imagen
        stof(string(argv[3])) // pocentaje al cual fue entrenado
    );
};
