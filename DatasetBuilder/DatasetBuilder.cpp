#include "pch.h"
#include "cmdlineopt.h"
#include "Dataset.h"


int main(int argc, const char* argv[]) {
    try {
        // Opciones de línea de comando
        // --path=C:\Repositories\CementCrack\Prueba --prefix=64x64 --size=64 --verbose 
        CmdLineOpt::CmdLineOpt(argc, argv);

        Dataset(
            CmdLineOpt::dataset_path,
            CmdLineOpt::dataset_prefix,
            CmdLineOpt::image_size
        );

        return 0;
    }
    catch (std::exception e) {
        std::cerr << "Error: " << e.what();
    }
};

