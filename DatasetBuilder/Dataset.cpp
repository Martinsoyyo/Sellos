#include "pch.h"
#include "Dataset.h"
#include "cmdlineopt.h"

auto Dataset::get_file_or_directory_structure(
    const size_t& TYPE,
    const std::string& PATH,
    const std::string& EXT = "")
{
    std::vector<std::string> VEC;
    WIN32_FIND_DATA data;
    HANDLE hFind = FindFirstFile(std::string(PATH + DSEP + "*." + EXT).c_str(), &data);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (data.dwFileAttributes == TYPE) VEC.push_back(data.cFileName);
        } while (FindNextFile(hFind, &data));
    }

    FindClose(hFind);
    return VEC;
};

auto Dataset::get_image_from_directory(
    const std::string& PATH,
    const std::string& EXT)
{
    std::string THUMBNAILS_DIRECTORY = PATH + DSEP + std::to_string(CmdLineOpt::image_size) + "x" + std::to_string(CmdLineOpt::image_size) + " IMG";
    _mkdir(THUMBNAILS_DIRECTORY.c_str());
    size_t count = 0;

    std::vector<torch::Tensor> IMG;
    auto FILES = get_file_or_directory_structure(FILE_ATTRIBUTE_ARCHIVE, PATH , EXT);
    for (const auto& filename : FILES) {
        /*
        //---------------------------------------------------------------------
        // Pre-tratamiento de cada imagen con OpenCV.
        cv::Mat aux = cv::imread(PATH + DSEP + filename);
        std::vector<cv::Mat> rgb;
        cv::resize(aux, aux, cv::Size(CmdLineOpt::image_size, CmdLineOpt::image_size));
        cv::split(aux, rgb);
        //---------------------------------------------------------------------
        auto m_images = torch::zeros({ 3,CmdLineOpt::image_size,CmdLineOpt::image_size }, torch::kByte);
        m_images[0] = torch::from_blob(rgb[0].data, { CmdLineOpt::image_size, CmdLineOpt::image_size }, torch::kByte);
        m_images[1] = torch::from_blob(rgb[1].data, { CmdLineOpt::image_size, CmdLineOpt::image_size }, torch::kByte);
        m_images[2] = torch::from_blob(rgb[2].data, { CmdLineOpt::image_size, CmdLineOpt::image_size }, torch::kByte);

        IMG.push_back(m_images);
        */

        cv::Mat src = cv::imread(PATH + DSEP + filename);
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

        //Cropping la imagen centrada
        /*
        uint32_t X0 = 0.5 * (src.size().width - src.size().height);
        cv::Mat ROI(src, cv::Rect(X0, 0, src.size().height, src.size().height));
        ROI.copyTo(src);
        */

        //Partir la imagen cuadrada en 9.
        /*
        uint32_t X = 0, Y = 0;
        uint32_t DELTA = src.size().height / 3;
        for (int k = 0; k < 3; k++) 
        {
            for (int j = 0; j < 3; j++) 
            {
                cv::Mat cropped_src;
                cv::Mat ROI(src, cv::Rect(X, Y, DELTA, DELTA));
                cv::imwrite(THUMBNAILS_DIRECTORY + DSEP + std::to_string(count++) + ".jpg", ROI);
                X += DELTA;
            }
            X = 0;
            Y += DELTA;
        }
        */

        cv::resize(src, src, cv::Size(CmdLineOpt::image_size, CmdLineOpt::image_size));
        if (CmdLineOpt::augmentation == true) {
            for (int k = 0; k < 2; k++) {
                for (int j = 0; j < 2; j++) {
                    for (auto i = 0; i < 4; i++) {
                        auto m_images = torch::from_blob(src.data, { CmdLineOpt::image_size, CmdLineOpt::image_size }, torch::kByte);
                        cv::imwrite(THUMBNAILS_DIRECTORY + DSEP + std::to_string(count++) + ".jpg", src); //graba jpgs al disco
                        IMG.push_back(m_images.clone());

                        cv::rotate(src, src, 0);
                    }
                    cv::flip(src, src, 0); //flip horizontal
                }
                cv::flip(src, src, 1); //flip vertical
            }

        }
        else {
            auto m_images = torch::from_blob(src.data, { CmdLineOpt::image_size, CmdLineOpt::image_size }, torch::kByte);
            cv::imwrite(THUMBNAILS_DIRECTORY + DSEP + std::to_string(count++) + ".jpg", src);
            IMG.push_back(m_images.clone());
        }

        //checkeo que este guardando lo que dice..
        /*cv::Mat cv_mat = cv::Mat(CmdLineOpt::image_size, CmdLineOpt::image_size, CV_8U);
        for (int i = 0; i < 8; i++) {
            //std::cout << IMG[i] << std::endl;
            std::memcpy(cv_mat.data, IMG[i].data_ptr(), 64 * 64);
            std::cout << "";
        };
        */
    
    }

    //IMG.push_back(torch::randn({ 64,64 }));
    return torch::stack(torch::TensorList(IMG), 0);
};

Dataset::Pair Dataset::proccesing_data(
    const std::string& PATH,
    const std::string& EXT,
    const uint32_t& SIZE)
{
    std::vector<torch::Tensor> IMG;
    std::vector<torch::Tensor> TRG;

    auto DIR = get_file_or_directory_structure(FILE_ATTRIBUTE_DIRECTORY, PATH);

    if (CmdLineOpt::verbose) std::cout << "Procesando Directorios." << std::endl;
    int count = 0;
    for (const auto& dirname : DIR) {
        if (dirname != "." && dirname != "..") {
            std::string STR = PATH + DSEP + dirname;
            if (CmdLineOpt::verbose) std::cout << STR << std::endl;
            IMG.push_back(get_image_from_directory(STR, "jpg"));
            TRG.push_back(torch::full({ IMG[count].size(0) }, count).to(at::kByte));
            count++;
        }
    };

    return { torch::cat(torch::TensorList(IMG)) ,torch::cat(torch::TensorList(TRG)) };
}

Dataset::Dataset(
    const std::string& ROOT_FOLDER,
    const std::string& PREFIX_FN,
    uint32_t IMAGE_SIZE)
{
    Dataset::Pair T = proccesing_data(ROOT_FOLDER , "jpg", IMAGE_SIZE);

    if (CmdLineOpt::verbose) std::cout << "Mezclando y Guardando." << std::endl;
    auto IDX = torch::randperm(T.first.size(0)).to(torch::kLong);

    if (CmdLineOpt::verbose) {
        std::cout << "Guardando IMAGE en.";
        std::cout << IMG_FNAME(ROOT_FOLDER, PREFIX_FN) << std::endl;
        std::cout << "Guardando TARGET en.";
        std::cout << TRG_FNAME(ROOT_FOLDER, PREFIX_FN) << std::endl;
    }

    torch::save(torch::index_select(T.first, 0, IDX), IMG_FNAME(ROOT_FOLDER, PREFIX_FN));
    torch::save(torch::index_select(T.second, 0, IDX), TRG_FNAME(ROOT_FOLDER, PREFIX_FN));
}
