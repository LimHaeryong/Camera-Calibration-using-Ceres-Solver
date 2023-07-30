#include "Calibration/Calibration.h"

int main(int argc, char **argv)
{
    if(argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " [image_directory] [intrinsic.yaml]" << std::endl;
        return 1;
    }

    std::string image_directory = argv[1];
    YAML::Node config = YAML::LoadFile(argv[2]);

    Calibration calibration(config);

    std::vector<std::string> image_paths;

    cv::glob(image_directory + "/*.jpg", image_paths);
    cv::glob(image_directory + "/*.png", image_paths);

    int count = 0;
    for(const auto& image_path : image_paths)
    {
        cv::Mat src = cv::imread(image_path, cv::IMREAD_ANYCOLOR);
        cv::Mat dst;
        calibration.undistort(src, dst);
        cv::imshow("undistorted image", dst);
        cv::waitKey(100);
        cv::imwrite("../result/" + std::to_string(count++) + ".jpg", dst);
    }

    return 0;
}
