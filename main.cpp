#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


class Leaf {
public:
	Leaf(Mat image, Size sizeBeforeCrop) :
		image(image),
		sizeBeforeCrop(sizeBeforeCrop)
	{ }

	Mat image;
	Size sizeBeforeCrop;
};

Mat cropImage(Mat img) {

	const int height = 26; // Średnia wysokość dużej litery na zbiorze danych.

	bool rowEmpty;

	int fstRow = img.rows, lastRow = -1;
	int fstCol = img.cols, lastCol = -1;

	for (int y = 0; y < img.rows; ++y) {
		rowEmpty = true;
		for (int x = 0; x < img.cols; ++x) {
			if (img.at<uchar>(y, x) != 255) {
				fstCol   = min( fstCol, x);
				lastCol  = max(lastCol, x);
				rowEmpty = false;
			}
		}

		if (!rowEmpty) {
			fstRow  = min( fstRow, y);
			lastRow = max(lastRow, y);
		}
	}

	// Mat res = img(Rect(fstCol, fstRow, lastCol - fstCol + 1, lastRow - fstRow + 1));
	// cerr << "fstRow: " << fstRow << ", fstCol" << fstCol << ", lastRow: " << lastRow << ", lastCol: " << lastCol << endl;
	// cerr << "width: " << lastCol - fstCol << endl;
	// cerr << "height: " << lastRow - fstRow << endl;
	//	imshow("img", res);
	//	waitKey(0);

	Mat res = 255 - Mat(height, lastCol - fstCol + 1, CV_8UC1);

	int h = min(height, lastRow - fstRow + 1);

//	int top = (h <= height) ?

	// |    |      ***********************   |    |
	//	res()

	return res;
}

shared_ptr<Leaf> createLeaf(Mat &img) {

	return shared_ptr<Leaf>(new Leaf(img, Size(img.cols, img.rows)));
}

vector<shared_ptr<Leaf>> openImages() {
	vector<shared_ptr<Leaf>> res;

	int i = 0;
	cerr << "Opening images.";

	while (true) {
		stringstream stream;
		stream << "data/" << i++ << ".png";
		std::string name = stream.str();

		if (i % 250 == 0)
			cerr << ".";

		Mat tmp = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
		if (tmp.empty())
			break;

		res.push_back(createLeaf(tmp));
	}

	cerr << endl;

	return res;
}

void processImage(shared_ptr<Leaf> img) {

	//	cv::resize(img, img, Size(8, 13));

	if (img->image.rows >= 6 && img->image.rows <= 9) {

		// imshow("img", img->image);
		// waitKey(500);

		cerr << "height: " << img->sizeBeforeCrop.height << " -> " << img->image.rows << endl;
	}
}

int main()
{
	namedWindow("img", CV_WINDOW_NORMAL);

	vector<shared_ptr<Leaf>> images = openImages();
	vector<int> lengths, heights;

	cerr << "Computing lengths";

	int lsum = 0, hsum = 0;
	for (int i = 0; i < images.size(); ++i) {
		if (i % 100 == 0)
			cerr << ".";

		while (lengths.size() <= images[i]->image.cols)
			lengths.push_back(0);

		while (heights.size() <= images[i]->image.rows)
			heights.push_back(0);

		++lengths[images[i]->image.cols];
		++heights[images[i]->image.rows];

		lsum += images[i]->image.cols;
		hsum += images[i]->image.rows;

		processImage(images[i]);
	}

	cerr << endl;

	std::cout << "Lengths" << std::endl;
	for (size_t i = 0; i < lengths.size(); ++i)
		std::cout << i << "\t" << lengths[i] << std::endl;

	std::cout << "Heights" << std::endl;
	for (size_t i = 0; i < heights.size(); ++i)
		std::cout << i << "\t" << heights[i] << std::endl;

	cout << "h avg" << (float) hsum / images.size() << endl;
	cout << "w avg" << (float) lsum / images.size() << endl;

	return 0;
}
