#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class ImageData {
public:
	ImageData(Mat img) : img(img) { }

	Mat    img;
	Point  massCentre;
	string fileName;
};

float getDistance(const ImageData *d1, const ImageData *d2) {

	const int maxSizeDiff = 4;
	if (abs(d1->img.cols - d2->img.cols) > maxSizeDiff || abs(d1->img.rows - d2->img.rows) > maxSizeDiff)
		return 128.0;

	Point tl = Point(
		-min(d1->massCentre.x, d2->massCentre.x),
		-min(d1->massCentre.y, d2->massCentre.y)
	);
	Point br = Point(
		min(d1->img.cols - d1->massCentre.x, d2->img.cols - d2->massCentre.x),
		min(d1->img.rows - d1->massCentre.y, d2->img.rows - d2->massCentre.y)
	);

	int sum = 0;
	int weights = 0;
	for (int dy = tl.y; dy <= br.y; ++dy) {
		for (int dx = tl.x; dx <= br.x; ++dx) {
			uchar a = d1->img.at<uchar>(d1->massCentre.y + dy, d1->massCentre.x + dx);
			uchar b = d2->img.at<uchar>(d2->massCentre.y + dy, d2->massCentre.x + dx);
			sum     += min(510 - a - b, 255) * abs(a - b);
			weights += min(510 - a - b, 255);
		}
	}

	return (float)sum / weights;
}

void cropImage(Mat &img, Mat &res, Point &massCentre) {

	massCentre = Point(0, 0);
	int weightsSum = 0;
	bool rowEmpty;
	int fstRow = img.rows, lastRow = -1;
	int fstCol = img.cols, lastCol = -1;

	for (int y = 0; y < img.rows; ++y) {
		rowEmpty = true;
		for (int x = 0; x < img.cols; ++x) {
			uchar val = img.at<uchar>(y, x);

			if (val != 255) {
				fstCol   = min( fstCol, x);
				lastCol  = max(lastCol, x);
				rowEmpty = false;
				massCentre += Point(x, y) * (255 - val);
				weightsSum += (255 - val);
			}
		}

		if (!rowEmpty) {
			fstRow  = min( fstRow, y);
			lastRow = max(lastRow, y);
		}
	}

	massCentre = Point((float)massCentre.x / weightsSum - fstCol, (float)massCentre.y / weightsSum - fstRow);
	res = img(Rect(fstCol, fstRow, lastCol - fstCol + 1, lastRow - fstRow + 1));
}

void preprocessImage(ImageData *data) {
	// Erode

	//	const int erosionSize = 1;
	//	static Mat element = getStructuringElement(
	//		MORPH_ELLIPSE,
	//		Size( 2 * erosionSize + 1, 2 * erosionSize + 1 ),
	//		Point( erosionSize, erosionSize )
	//	);
	//	erode (img, img, element);
	//	dilate(img, img, element);

	// Blur
	GaussianBlur(data->img, data->img, Size(3, 3), 0);

	// Crop
	cropImage(data->img, data->img, data->massCentre);
}

void openImages(vector<ImageData *> &images) {
	images.clear();

	int i = 0;
	cerr << "Opening and preprocessing images:" << endl;

	while (true) {
		stringstream stream;
		stream << "data/" << i++ << ".png";
		std::string name = stream.str();

		if (i % 250 == 0)
			cerr << "\r" << i;

		Mat tmp = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
		if (tmp.empty())
			break;

		ImageData *data = new ImageData(tmp);
		preprocessImage(data);
		images.push_back(data);
	}

	cerr << "\rOpened " << images.size() << " images " << endl;
}

void dbscan(const vector<ImageData *>& data) {
	vector<vector<char>> distances;
	vector<bool> visited(data.size(), false);

	cerr << "Computing pairwise distances:" << endl;

	// Compute distance matrix
	int counter = 0;
	int p1 = data.size() * (data.size() + 1) / 200;
	for (int i = 0; i < data.size(); ++i) {
		distances.push_back(vector<char>());

		for (int j = 0; j < i; ++j, ++counter) {
			distances[i].push_back(getDistance(data[i], data[j]));

			if (counter % p1 == 0)
				cerr << counter / p1 << "%\r";
		}
	}

	cerr << "Done" << endl;
}

int main()
{
	vector<ImageData *> data;
	openImages(data);
	dbscan(data);

	return 0;
}

//int main()
//{

//	vector<ImageData *> data;
//	openImages(data);

//	vector<int> badLengths, goodLengths;

//	cerr << "Images opened" << endl;

//	vector<int> dists;
//	for (int i = 0; i < 256; ++i) {
//		dists.push_back(0);
//		badLengths.push_back(0);
//		goodLengths.push_back(0);
//	}

//	for (int i = 0; i < data.size(); ++i) {
//		float minDist = 1000;
//		float dist;
//		int minIdx = -1;

//		if (i % 100 == 0)
//			cerr << "Processed " << i << " out of " << data.size() << endl;

//		for (int j = 0; j < data.size(); ++j) {
//			dist = getDistance(data[i], data[j]);
//			++dists[(int)dist];
//			if (dist < minDist && i != j) {
//				minDist = dist;
//				minIdx = j;
//			}
//		}
//		imshow("img_a", data[i]->img);
//		imshow("img_b", data[minIdx]->img);
//		cerr << "distance: " << minDist;
//		char key = waitKey(0);
//		if (key == 'g')
//			++goodLengths[(int)minDist];
//		else if (key == 'q')
//			break;
//		else
//			++badLengths[(int)minDist];
//	}
////	for (int i = 0; i < 256; ++i)
////		cerr << i << "\t" << dists[i] << endl;

//	for (int i = 0; i < 256; ++i)
//		cerr << "i" << "\t" << badLengths[i] << endl;

//	for (int i = 0; i < 256; ++i)
//		cerr << "i" << "\t" << goodLengths[i] << endl;

//	return 0;
//}

