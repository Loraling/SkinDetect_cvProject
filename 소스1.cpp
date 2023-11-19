#define DLIB_JPEG_SUPPORT

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <iostream>

using namespace dlib;
using namespace std;
using namespace cv;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    std::vector<pair<int, int>> left_eyebrows, right_eyebrows, lip;
    if (argc != 3)
    {
        cout << "두 개의 파일을 인자로 넣어야 합니다." << endl;
        return 1;
    }

    frontal_face_detector detector = get_frontal_face_detector();

    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

    shape_predictor sp2;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp2;

    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;


    std::vector<dlib::rectangle> faces;

    for (int i = 1; i <= 2; i++) {
        matrix<rgb_pixel> img;
        load_image(img, argv[i]);
        auto face = detector(img)[0];
        auto shape = sp(img, face);

        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape, 150, 0.1), face_chip);
        faces.push_back(move(face));

        std::vector<full_object_detection> shapes;
        full_object_detection shape = sp2(img, face2);
        for (int i = 17; i <= 21; i++) { // (사진 상) 왼쪽 눈썹
            shape.part(17)
        }
    }

    Mat original_image, face_image, ycrcb_image, hsv_image, y_skin_msk, h_skin_msk, y_skin, h_skin;
    Rect face_rect(faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height());
    original_image = imread(argv[1], IMREAD_COLOR);
    face_image = original_image(face_rect);

    cvtColor(face_image, ycrcb_image, COLOR_BGR2YCrCb);
    inRange(ycrcb_image, Scalar(0, 133, 77), Scalar(255, 173, 127), y_skin_msk);
    cvtColor(face_image, hsv_image, COLOR_BGR2HSV);
    inRange(hsv_image, Scalar(0, 48, 80), Scalar(20, 255, 255), h_skin_msk);

    bitwise_and(face_image, face_image, y_skin, y_skin_msk);
    bitwise_and(face_image, face_image, h_skin, h_skin_msk);

    imshow("face", face_image);
    imshow("y skin", y_skin);
    imshow("h skin", h_skin);
    
    waitKey(0);

    destroyAllWindows();

}
catch (std::exception& e)
{
    cout << e.what() << endl;
}
