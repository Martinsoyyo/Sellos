#pragma once
#include "pch.h"

#define DSEP "\\"
#define IMG_FNAME(ROOT_FOLDER,PREFIX_FN) ROOT_FOLDER + DSEP + PREFIX_FN + "IMAGES.tensor"
#define TRG_FNAME(ROOT_FOLDER,PREFIX_FN) ROOT_FOLDER + DSEP + PREFIX_FN + "TARGET.tensor"
using namespace std;

class Dataset
{
public:
    using Pair = std::pair<torch::Tensor, torch::Tensor>;

    Dataset(const std::string& ROOT_FOLDER, const std::string& PREFIX_FN, uint32_t IMAGE_SIZE);

    auto get_file_or_directory_structure(const size_t& TYPE,const std::string& PATH,const std::string& EXT);
    auto get_image_from_directory(const std::string& PATH, const std::string& EXT);
    Dataset::Pair proccesing_data(const std::string& PATH, const std::string& EXT, const uint32_t& SIZE);
};

