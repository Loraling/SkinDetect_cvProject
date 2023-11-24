#define DLIB_JPEG_SUPPORT
#define DLIB_PNG_SUPPORT

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
    if (argc != 3)
    {
        cout << "�� ���� ������ ���ڷ� �־�� �մϴ�." << endl;
        return 1;
    }

    frontal_face_detector detector = get_frontal_face_detector();

    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

    shape_predictor sp2;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp2;

    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    std::vector<int> score;


    for (int i = 1; i <= 2; i++) {
        matrix<rgb_pixel> img;
        load_image(img, argv[i]);
        auto face = detector(img)[0];
        auto shape = sp(img, face);

        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape, 150, 0.1), face_chip);


        std::vector<cv::Point> left_eyebrows, right_eyebrows, nose, left_eye, right_eye, lip;
        shape = sp2(img, face);

        for (int i = 17; i <= 21; i++) { // (���� ��) ���� ����
            Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
            left_eyebrows.push_back(p);
        }

        for (int i = 22; i <= 26; i++) { // (���� ��) ������ ����
            Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
            right_eyebrows.push_back(p);
        }

        for (int i = 31; i <= 35; i++) { // ��
            if (i == 31) {
                Point n = Point(shape.part(27).x() - face.left(), shape.part(27).y() - face.top());
                nose.push_back(n);
            }
            Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
            nose.push_back(p);
        }
        nose[1].x -= (2 * nose[1].x - nose[3].x > 0) ? nose[3].x - nose[1].x : 0;
        nose[5].x += (2 * nose[5].x - nose[3].x < face.width()) ? nose[5].x - nose[3].x : 0;

        for (int i = 36; i <= 41; i++) { // (���� ��) ���� ��
            Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
            left_eye.push_back(p);
        }
        left_eye[0].x -= (2 * left_eye[0].x - left_eye[1].x > 0) ? left_eye[1].x - left_eye[0].x : 0;
        left_eye[3].x += (2 * left_eye[3].x - left_eye[2].x < face.width()) ? left_eye[3].x - left_eye[2].x : 0;
        left_eye[1].y -= (2 * left_eye[1].y - left_eye[5].y > 0) ? left_eye[5].y - left_eye[1].y : 0;
        left_eye[5].y += (2 * left_eye[5].y - left_eye[1].y < face.height()) ? left_eye[5].y - left_eye[1].y : 0;
        left_eye[2].y -= (2 * left_eye[2].y - left_eye[4].y > 0) ? left_eye[4].y - left_eye[2].y : 0;
        left_eye[4].y += (2 * left_eye[4].y - left_eye[2].y < face.height()) ? left_eye[4].y - left_eye[2].y : 0;

        for (int i = 42; i <= 47; i++) { // (���� ��) ������ ��
            Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
            right_eye.push_back(p);
        }
        right_eye[0].x -= (2 * right_eye[0].x - right_eye[1].x > 0) ? right_eye[1].x - right_eye[0].x : 0;
        right_eye[3].x += (2 * right_eye[3].x - right_eye[2].x < face.width()) ? right_eye[3].x - right_eye[2].x : 0;
        right_eye[1].y -= (2 * right_eye[1].y - right_eye[5].y > 0) ? right_eye[5].y - right_eye[1].y : 0;
        right_eye[5].y += (2 * right_eye[5].y - right_eye[1].y < face.height()) ? right_eye[5].y - right_eye[1].y : 0;
        right_eye[2].y -= (2 * right_eye[2].y - right_eye[4].y > 0) ? right_eye[4].y - right_eye[2].y : 0;
        right_eye[4].y += (2 * right_eye[4].y - right_eye[2].y < face.height()) ? right_eye[4].y - right_eye[2].y : 0;

        for (int i = 48; i <= 59; i++) { // �Լ�
            Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
            lip.push_back(p);
        }
        lip[0].x -= (2 * lip[0].x - lip[1].x > 0) ? (lip[1].x - lip[0].x) / 2 : 0;
        lip[6].x += (2 * lip[6].x - lip[5].x < face.width()) ? (lip[6].x - lip[5].x) / 2 : 0;

        Mat original_image, face_image, ycrcb_image, hsv_image, skin_msk, skin;
        Rect face_rect(face.left(), face.top(), face.width(), face.height());
        original_image = imread(argv[i], IMREAD_COLOR);

        if (face_rect.x < 0) face_rect.x = 0;
        else if (face_rect.x > original_image.cols) face_rect.x = original_image.cols - 1;
        if (face_rect.y < 0) face_rect.y = 0;
        else if (face_rect.x > original_image.rows) face_rect.y = original_image.rows - 1;
        if (face_rect.x + face_rect.width > original_image.cols) face_rect.width = 1;
        if (face_rect.y + face_rect.height > original_image.rows) face_rect.height = 1;

        face_image = original_image(face_rect);

        cvtColor(face_image, ycrcb_image, COLOR_BGR2YCrCb);
        inRange(ycrcb_image, Scalar(0, 133, 77), Scalar(255, 173, 127), skin_msk);

        fillPoly(skin_msk, left_eyebrows, Scalar(0, 0, 0));
        fillPoly(skin_msk, right_eyebrows, Scalar(0, 0, 0));
        fillPoly(skin_msk, nose, Scalar(0, 0, 0));
        fillPoly(skin_msk, left_eye, Scalar(0, 0, 0));
        fillPoly(skin_msk, right_eye, Scalar(0, 0, 0));
        fillPoly(skin_msk, lip, Scalar(0, 0, 0));

        polylines(skin_msk, left_eyebrows, 1, Scalar(0, 0, 0), 5);
        polylines(skin_msk, right_eyebrows, 1, Scalar(0, 0, 0), 5);
        polylines(skin_msk, nose, 1, Scalar(0, 0, 0), 5);
        polylines(skin_msk, left_eye, 1, Scalar(0, 0, 0), 5);
        polylines(skin_msk, right_eye, 1, Scalar(0, 0, 0), 5);
        polylines(skin_msk, lip, 1, Scalar(0, 0, 0), 5);

        erode(skin_msk, skin_msk, Mat::ones(Size(3, 3), CV_8UC1));

        bitwise_and(face_image, face_image, skin, skin_msk);
        resize(skin, skin, Size(480, 480));

        std::vector<int> b, g, r;
        
        for (int i = 0; i < skin.rows; i++) {
            for (int j = 0; j < skin.cols; j++) {
                Vec3b& pixel = skin.at<Vec3b>(i, j);
                if (pixel[0] != 0 && pixel[1] != 0 && pixel[2] != 0) {
                    b.push_back(pixel[0]);
                    g.push_back(pixel[1]);
                    r.push_back(pixel[2]);
                }
            }
        }

        double b_avg = 0.0, g_avg = 0.0, r_avg = 0.0;
        for (int i = 0; i < b.size(); i++) {
            b_avg += b[i];
            g_avg += g[i];
            r_avg += r[i];
        }
        b_avg /= b.size(); g_avg /= g.size(); r_avg /= r.size();
        double b_mV = 1 / b_avg, g_mV = 1 / g_avg, r_mV = 1 / r_avg;
        double max_mV = max(b_mV, max(g_mV, r_mV));

        double b_scaled = b_mV / max_mV, g_scaled = g_mV / max_mV, r_scaled = r_mV / max_mV;

        for (int i = 0; i < skin.rows; i++) {
            for (int j = 0; j < skin.cols; j++) {
                Vec3b& pixel = skin.at<Vec3b>(i, j);
                if (pixel[0] != 0 && pixel[1] != 0 && pixel[2] != 0) {
                    pixel[0] = pixel[0] * b_scaled;
                    pixel[1] = pixel[1] * g_scaled;
                    pixel[2] = pixel[2] * r_scaled;
                }
            }
        }

        Mat skin_lab;
        cvtColor(skin, skin_lab, COLOR_BGR2Lab);
        std::vector<Mat> channels(3);
        split(skin_lab, channels);
        Mat skin_ac(skin_lab.size(), CV_8UC1);

        double min_a, max_a;
        minMaxLoc(channels[1], &min_a, &max_a);
        max_a -= 128;

        for (int i = 0; i < channels[1].rows; i++) {
            for (int j = 0; j < channels[1].cols; j++) {
                uchar pixel = channels[1].at<uchar>(i, j);
                //cout << (double)pixel - 128 << endl;
                if (((double)pixel - 128) / max_a > 0.5) skin_ac.at<uchar>(i, j) = 255;
                else skin_ac.at<uchar>(i, j) = 0;
            }
        }

        score.push_back(countNonZero(skin_ac));

        resize(face_image, face_image, Size(480, 480));

        Mat skin_ac_weight = skin_ac.clone(), face_result=face_image.clone();
        cvtColor(skin_ac, skin_ac_weight, COLOR_GRAY2BGR);
        addWeighted(face_image, 0.5, skin_ac_weight, 0.5, 0.0, face_result);
        imshow("face_image", face_result);
        imshow("skin_AC", skin_ac);

        waitKey(0);

        destroyAllWindows();
    }
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}