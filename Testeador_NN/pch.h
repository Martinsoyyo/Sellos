#include <torch/torch.h>
#include <torch/script.h>

#include <stdint.h>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <stdlib.h>
#include <stdio.h>
#include <direct.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define OS_WINDOWS (defined(_WIN32) || defined(_WIN64)...)
#ifdef OS_WINDOWS
#include <windows.h>
#endif


using namespace std;
using namespace cv;
using namespace torch::nn;