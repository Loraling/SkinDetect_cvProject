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

// 얼굴 영상에서 눈썹, 눈, 코, 입 위치 검출
Mat make_face_mask(dlib::input_rgb_image::input_type img, dlib::rectangle face);

// 얼굴 영상에서 색상 속성 검사를 통해 피부 영역 검출
Mat make_skin_mask(Mat img, Rect face);

// 피부 영상에서 여드름 수치 파악
Mat acne_detect(Mat skin_arg);


int main(int argc, char** argv) try
{
    if (argc != 3)
    {
        cout << "두 개의 파일을 인자로 넣어야 합니다." << endl;
        return 1;
    }

    frontal_face_detector detector = get_frontal_face_detector();

    std::vector<int> score;


    for (int i = 1; i <= 2; i++) {
        matrix<rgb_pixel> img;
        load_image(img, argv[i]);
        auto face = detector(img)[0];

        // 얼굴 ROI 영역이 참조 불가능한 경우 속성치 수정
        if (face.left() < 0) face.set_left(0);
        else if (face.left() > img.nc()) face.set_left(img.nc() - 1);
        if (face.top() < 0) face.set_top(0);
        else if (face.top() > img.nr()) face.set_top(img.nr() - 1);
        if (face.left() + face.width() > img.nc()) face.set_right(img.nc() - 1);
        if (face.top() + face.height() > img.nr()) face.set_bottom(img.nr() - 1);


        // dlib 바탕의 자료형을 OpenCV 자료형으로 전환
        Rect face_rect(face.left(), face.top(), face.width(), face.height());
        Mat original_image = imread(argv[i], IMREAD_COLOR);


        // 얼굴 영상에서 피부 추출
        Mat face_mask = make_face_mask(img, face);
        Mat skin_mask = make_skin_mask(original_image, face_rect);
        bitwise_and(skin_mask, face_mask, skin_mask);
        erode(skin_mask, skin_mask, Mat::ones(Size(3, 3), CV_8UC1));

        Mat face_image = original_image(face_rect);

        Mat skin;

        cvtColor(skin_mask, skin_mask, COLOR_GRAY2BGR);
        bitwise_and(face_image, skin_mask, skin);

        // 출력 테스트
        /*
        resize(skin, skin, Size(480, 480));
        resize(face_image, face_image, Size(480, 480));
        imshow("skin_mask", skin);*/

        // 여드름 검출 후 수치화
        Mat acne_mask = acne_detect(skin);
        score.push_back(countNonZero(acne_mask));

        Mat acne_weight = acne_mask.clone(), face_result=face_image.clone();
        cvtColor(acne_mask, acne_weight, COLOR_GRAY2BGR);
        addWeighted(face_image, 0.5, acne_weight, 0.5, 0.0, face_result);
        imshow("face_result", face_result);

        waitKey(0);
        destroyAllWindows();
    }

    cout << argv[1] << " 파일의 피부 트러블 수치 : " << score[0] << endl;
    cout << argv[2] << " 파일의 피부 트러블 수치 : " << score[1] << endl;
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}

Mat make_face_mask(dlib::input_rgb_image::input_type img, dlib::rectangle face) {
    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

    std::vector<cv::Point> left_eyebrows, right_eyebrows, nose, left_eye, right_eye, lip;

    // 사전에 생성된 모델을 통해 얼굴 랜드마킹
    auto shape = sp(img, face);

    for (int i = 17; i <= 21; i++) { // (사진 상) 왼쪽 눈썹
        Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
        left_eyebrows.push_back(p);
    }

    for (int i = 22; i <= 26; i++) { // (사진 상) 오른쪽 눈썹
        Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
        right_eyebrows.push_back(p);
    }

    for (int i = 31; i <= 35; i++) { // 코
        if (i == 31) {
            Point n = Point(shape.part(27).x() - face.left(), shape.part(27).y() - face.top());
            nose.push_back(n);
        }
        Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
        nose.push_back(p);
    }
    // 특정 얼굴 요소를 완전히 가리기 위해 좌표값 수정
    nose[1].x -= (2 * nose[1].x - nose[3].x > 0) ? nose[3].x - nose[1].x : 0;
    nose[5].x += (2 * nose[5].x - nose[3].x < face.width()) ? nose[5].x - nose[3].x : 0;

    for (int i = 36; i <= 41; i++) { // (사진 상) 왼쪽 눈
        Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
        left_eye.push_back(p);
    }
    left_eye[0].x -= (2 * left_eye[0].x - left_eye[1].x > 0) ? left_eye[1].x - left_eye[0].x : 0;
    left_eye[3].x += (2 * left_eye[3].x - left_eye[2].x < face.width()) ? left_eye[3].x - left_eye[2].x : 0;
    left_eye[1].y -= (2 * left_eye[1].y - left_eye[5].y > 0) ? left_eye[5].y - left_eye[1].y : 0;
    left_eye[5].y += (2 * left_eye[5].y - left_eye[1].y < face.height()) ? left_eye[5].y - left_eye[1].y : 0;
    left_eye[2].y -= (2 * left_eye[2].y - left_eye[4].y > 0) ? left_eye[4].y - left_eye[2].y : 0;
    left_eye[4].y += (2 * left_eye[4].y - left_eye[2].y < face.height()) ? left_eye[4].y - left_eye[2].y : 0;

    for (int i = 42; i <= 47; i++) { // (사진 상) 오른쪽 눈
        Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
        right_eye.push_back(p);
    }
    right_eye[0].x -= (2 * right_eye[0].x - right_eye[1].x > 0) ? right_eye[1].x - right_eye[0].x : 0;
    right_eye[3].x += (2 * right_eye[3].x - right_eye[2].x < face.width()) ? right_eye[3].x - right_eye[2].x : 0;
    right_eye[1].y -= (2 * right_eye[1].y - right_eye[5].y > 0) ? right_eye[5].y - right_eye[1].y : 0;
    right_eye[5].y += (2 * right_eye[5].y - right_eye[1].y < face.height()) ? right_eye[5].y - right_eye[1].y : 0;
    right_eye[2].y -= (2 * right_eye[2].y - right_eye[4].y > 0) ? right_eye[4].y - right_eye[2].y : 0;
    right_eye[4].y += (2 * right_eye[4].y - right_eye[2].y < face.height()) ? right_eye[4].y - right_eye[2].y : 0;

    for (int i = 48; i <= 59; i++) { // 입술
        Point p = Point(shape.part(i).x() - face.left(), shape.part(i).y() - face.top());
        lip.push_back(p);
    }
    lip[0].x -= (2 * lip[0].x - lip[1].x > 0) ? (lip[1].x - lip[0].x) / 2 : 0;
    lip[6].x += (2 * lip[6].x - lip[5].x < face.width()) ? (lip[6].x - lip[5].x) / 2 : 0;

    // 얼굴의 눈썹, 눈, 코, 입을 가리는 영상 생성
    Mat skin_mask(Size(face.width(), face.height()), CV_8UC1, Scalar(255, 255, 255));

    fillPoly(skin_mask, left_eyebrows, Scalar(0, 0, 0));
    fillPoly(skin_mask, right_eyebrows, Scalar(0, 0, 0));
    fillPoly(skin_mask, nose, Scalar(0, 0, 0));
    fillPoly(skin_mask, left_eye, Scalar(0, 0, 0));
    fillPoly(skin_mask, right_eye, Scalar(0, 0, 0));
    fillPoly(skin_mask, lip, Scalar(0, 0, 0));

    // 조금 더 두꺼운 외곽선으로 얼굴 요소들 주변의 명암이 짙은 영역 커버
    polylines(skin_mask, left_eyebrows, 1, Scalar(0, 0, 0), 5);
    polylines(skin_mask, right_eyebrows, 1, Scalar(0, 0, 0), 5);
    polylines(skin_mask, nose, 1, Scalar(0, 0, 0), 5);
    polylines(skin_mask, left_eye, 1, Scalar(0, 0, 0), 5);
    polylines(skin_mask, right_eye, 1, Scalar(0, 0, 0), 5);
    polylines(skin_mask, lip, 1, Scalar(0, 0, 0), 5);

    return skin_mask;
}

Mat make_skin_mask(Mat img, Rect face) {
    Mat face_image, ycrcb_image, skin_mask;

    // 얼굴 영역 영상
    face_image = img(face);

    // YCrCb 색상 체계에서 Cr과 Cb값의 범위를 통해 피부 영역 검출
    cvtColor(face_image, ycrcb_image, COLOR_BGR2YCrCb);
    inRange(ycrcb_image, Scalar(0, 133, 77), Scalar(255, 173, 127), skin_mask);

    return skin_mask;
}

Mat acne_detect(Mat skin_arg) {
    // 피부 영상을 CIE L*a*b* 색상 체계로 변환하여 여드름 영역을 검출
    // 검출 알고리즘은 하단의 논문을 참조
    // 박기홍, 노희성, "CIE L*a*b* 칼라 공간의 성분 영상 a*을 이용한 효과적인 여드름 검출," 디지털콘텐츠학회논문지 19, no. 7 (2018): 1397-1403, 10.9728/dcs.2018.19.7.1397

    // 피부 영상에 포함된 빛을 제거하기 위한 광 보상(light compensation) 과정
    std::vector<int> b, g, r;
    Mat skin = skin_arg.clone();

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
    // skin 인스턴스에 광 보상 완료

    // 해당 영상을 CIE L*a*b* 색상 체계로 변환 후, 성분 영상 a*을 이용하여 여드름 추정
    Mat skin_lab;
    cvtColor(skin, skin_lab, COLOR_BGR2Lab);
    std::vector<Mat> channels(3);
    split(skin_lab, channels);

    double min_a, max_a;
    minMaxLoc(channels[1], &min_a, &max_a);
    max_a -= 128;

    Mat skin_ac(skin_lab.size(), CV_8UC1);
    double threshold = 0.5; // 임계치는 임의로 설정. 향후 임계값 결정 알고리즘 등으로 개선 가능
    for (int i = 0; i < channels[1].rows; i++) {
        for (int j = 0; j < channels[1].cols; j++) {
            uchar pixel = channels[1].at<uchar>(i, j);
            if (((double)pixel - 128) / max_a > 0.5) skin_ac.at<uchar>(i, j) = 255;
            else skin_ac.at<uchar>(i, j) = 0;
        }
    }

    return skin_ac;
}