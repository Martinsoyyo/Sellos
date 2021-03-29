#include "pch.h"
#include "cmdlineopt.h"

namespace CmdLineOpt {

    bool augmentation           = false;
    bool verbose                = false;
    bool overwrite_dataset      = false;
    int image_size              = 64;;
    std::string dataset_path    = "";
    std::string dataset_prefix  = "";

    void CmdLineOpt::CmdLineOpt(int argc, const char* argv[]) {
        try {
            cxxopts::Options options(argv[0], " - example command line options");

            options
                .positional_help("[optional args]")
                .show_positional_help()
                //.parse_positional({ "first", "second", "last" })
                ;
            // --input=E:\data\CementCrack\5y9wdsg2zt-1 --prefix=32x32(0.15) --training_percentage=0.15 --size_x=32 --size_y=32 --verbose 
            options
                .allow_unrecognised_options()
                .add_options()
                ("h,help", "Print help")
                ("a,augmentation", "Uso de rotacion x4 y flip x2 de las imagenes dando 8x de la muestra", cxxopts::value<bool>(augmentation))
                ("s,size", "Tamano de las imagenes que se guardan en el tensor", cxxopts::value<int>(image_size))
                ("p,path", "Direccion donde se encuentran las IMAGENES", cxxopts::value<std::string>(dataset_path))
                ("prefix", "Prefijo para los nombres de los archivos de dataset '*.tensor' (ej. 'PREFIJO_TRAIN_IMAGES.tensor'", cxxopts::value<std::string>(dataset_prefix))
                ("v,verbose", "Informacion detallada mientras se ejecuta.", cxxopts::value<bool>(verbose))
                ;

            auto result = options.parse(argc, argv);

            if (result.count("augmentation")) std::cout << "Augmentation. <ON>" << std::endl;
            if (result.count("verbose")) std::cout << "Informacion detallada mientras se ejecuta. <ON>" << std::endl;
            if (result.count("path")) std::cout << "Direccion donde busca Imagenes. <" << dataset_path << ">" << std::endl;
            if (result.count("size")) std::cout << "Las Imagenes del tensor tienen dimension <" << image_size << "," << image_size << ">" << std::endl;
 
            if (result.count("dataset_prefix")) {
                dataset_prefix.append("_");
                std::cout << "Prefijo Dataset. <" << overwrite_dataset << ">" << std::endl;
            }

            if (result.count("help")) {
                std::cout << options.help({ "", "Group" }) << std::endl;
                exit(0);
            }
            std::cout << std::endl;

        }
        catch (const cxxopts::OptionException& e) {
            std::cout << "CmdLineOpt error: " << e.what() << std::endl;
            exit(1);
        }
    };
}