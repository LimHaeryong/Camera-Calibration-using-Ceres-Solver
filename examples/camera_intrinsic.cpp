#include "Calibration/Calibration.h"

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0] << " [image_directory] [optics/fisheye] [corners_width] [corners_height] [square(cm)]" << std::endl;
        return 1;
    }

    std::string image_directory = argv[1];
    std::string distortion_model = argv[2];
    int corners_width = std::stoi(argv[3]);
    int corners_height = std::stoi(argv[4]);
    double square = std::stod(argv[5]);

    Calibration calibration(image_directory, distortion_model, corners_width, corners_height, square);
    calibration.calibrate();
    calibration.save_yaml();

    return 0;
}