#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include "labels.h"
#include "thinning.h"

namespace fs = boost::filesystem;

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

//	imshow("img", d1->img);
//	waitKey(0);

	const int maxSizeDiff = 5;
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

	// Thinning
	Mat tmp;
	bitwise_not(data->img, tmp);
	thinning(tmp);
	bitwise_not(tmp, data->img);

	// Erode
	const int erosionSize = 1;
	static Mat element = getStructuringElement(
		MORPH_ELLIPSE,
		Size( 2 * erosionSize + 1, 2 * erosionSize + 1 ),
		Point( erosionSize, erosionSize )
	);
	erode (data->img, data->img, element);
	//	dilate(img, img, element);

//	imshow("img", data->img);
//	waitKey(0);

	// Blur
	GaussianBlur(data->img, data->img, Size(3, 3), 0);

	// Crop
	cropImage(data->img, data->img, data->massCentre);
}

void openImages(const fs::path &path, vector<ImageData *> &images) {

	cerr << "Opening and preprocessing images:" << endl;
	images.clear();
	int counter = 0;

	for (fs::directory_iterator it(path), eod; it != eod; ++it) {

		fs::path file = fs::absolute(*it);
		if (!file.has_extension() || file.extension().string() != ".png") {
			cerr << "Skipping file: \"" << file.string() << "\"" << endl;
			continue;
		}

		Mat tmp = imread(file.string(), CV_LOAD_IMAGE_GRAYSCALE);
		if (tmp.empty()) {
			cerr << "Could not load file \"" << file.string() << "\"" << endl;
			continue;
		}

		++counter;
		if (counter % 250 == 0)
			cerr << "\r" << counter;

		ImageData *data = new ImageData(tmp);
		data->fileName = file.filename().string();
		preprocessImage(data);
		images.push_back(data);
	}

	cerr << "\rOpened " << images.size() << " images " << endl;
}

void saveClusters(const fs::path &path, const vector<vector<ImageData *>> &clusters) {

	fs::ofstream out(path);
	size_t numberOfItems = 0;

	for (size_t i = 0; i < clusters.size(); ++i) {
		if (clusters[i].size() < 1)
			continue;

		out << clusters[i][0]->fileName;
		++numberOfItems;

		for (size_t j = 1; j < clusters[i].size(); ++j) {
			out << " " << clusters[i][j]->fileName;
			++numberOfItems;
		}

		out << endl;
	}

	out.close();
}

void assesClusters(const vector<vector<ImageData *>> &clusters) {

	cerr << "Assesing clusters." << endl;
	map<string, int> resultLabels;

	int i = 0;
	for (auto &cluster : clusters) {
		for (ImageData *element : cluster)
			resultLabels[element->fileName] = i;
		++i;
	}

	int e00 = 0, e01 = 0, e10 = 0, e11 = 0;

	for (auto &v1 : resultLabels) {
		for (auto &v2 : resultLabels) {

			if (v1.first == v2.first)
				continue;

			if (v1.second == v2.second) {
				if (labels[v1.first] == labels[v2.first])
					++e11;
				else
					++e10;
			} else {
				if (labels[v1.first] == labels[v2.first])
					++e01;
				else
					++e00;
			}
		}
	}

	cerr << "e00: " << e00 << ", e01: " << e01 << ", e10: " << e10 << ", e11: " << e11 << endl;
}

// Based on pseudocode from
// https://en.wikipedia.org/wiki/DBSCAN
void dbscanMain(const vector<ImageData *>& data, vector<vector<ImageData *>>& clusters) {

	const float eps    = 28.0;
	const float minPts = 6;

	vector<vector<float>> distances;
	vector<bool> visited;
	vector<bool> inCluster;

	visited.assign(data.size(), false);
	inCluster.assign(data.size(), false);
	clusters.clear();

	cerr << "Computing pairwise distances:" << endl;

	int counter = 0;
	int p1 = data.size() * (data.size() + 1) / 200;
	for (size_t i = 0; i < data.size(); ++i) {
		distances.push_back(vector<float>());

		for (size_t j = 0; j <= i; ++j, ++counter) {
			distances[i].push_back(getDistance(data[i], data[j]));

			if (counter % p1 == 0)
				cerr << counter / p1 << "%\r";
		}
	}

	for (size_t i = 0; i < data.size(); ++i)
		for (size_t j = i + 1; j < data.size(); ++j)
			distances[i].push_back(distances[j][i]);

	cerr << "Done" << endl;

	cerr << "Clustering graph" << endl;

	counter = 0;
	for (size_t P = 0; P < data.size(); ++P, ++counter) if (!visited[P]) {
		if (counter % 100 == 0)
			cerr << ".";
		visited[P] = true;
		vector<size_t> neighborPts;
		for (size_t i = 0; i < data.size(); ++i) if (distances[P][i] <= eps)
			neighborPts.push_back(i);

		if (neighborPts.size() < minPts)
			continue;

		clusters.push_back(vector<ImageData *>());
		vector<ImageData *> &C = clusters[clusters.size() - 1];

		C.push_back(data[P]);
		inCluster[P] = true;

		for (size_t P2i = 0; P2i < neighborPts.size(); ++P2i) {
			size_t P2 = neighborPts[P2i];
			if (!visited[P2]) {
				visited[P2] = true;

				vector<size_t> neighborPts2;
				for (size_t i = 0; i < data.size(); ++i) if (distances[P2][i] <= eps)
					neighborPts2.push_back(i);
				if (neighborPts2.size() >= minPts)
					for (size_t i = 0; i < neighborPts2.size(); ++i)
						if (!visited[neighborPts2[i]] || !inCluster[neighborPts2[i]])
							neighborPts.push_back(neighborPts2[i]);
			}

			if (!inCluster[P2]) {
				C.push_back(data[P2]);
				inCluster[P2] = true;
			}
		}
	}

	cerr << "Done." << endl;
}

void partitionMethod(const vector<ImageData *>& data, vector<vector<ImageData *>>&clusters) {

	vector<int> labels;
	int number = cv::partition(data, labels, [](ImageData *d1, ImageData *d2) {
		return getDistance(d1, d2) < 30.0;
	});

	cerr << "Number of cluster: " << number << endl;

	for (int i=  0; i < number; ++i)
		clusters.push_back(vector<ImageData *>());

	for (size_t i = 0; i < labels.size(); ++i) {
		clusters[labels[i]].push_back(data[i]);
	}

}

int main(int argc, char *argv[])
{
	if (argc != 3) {
		cerr
			<< "Usage:"  << endl
			<< argv[0] << " <input dir> <output file>" << endl
			<< "    <input dir>   - path to a directory containing *.png files" << endl
			<< "    <output file> - place where the output file should be created" << endl;
		return 1;
	}

	fs::path inputDirPath(argv[1]);
	fs::path outputPath(argv[2]);

	if (!fs::exists(inputDirPath)) {
		cerr << "Input folder \"" << inputDirPath << "\" does not exist" << endl;
		return 1;
	}

	if (!fs::is_directory(inputDirPath)) {
		cerr << "Specified input \"" << inputDirPath << "\" is not a directory" << endl;
		return 1;
	}

	if (fs::exists(outputPath)) {
		cerr << "Output file already exists and will be overwritten." << endl;
		// return 1;
	}

	vector<ImageData *>        data;
	vector<vector<ImageData*>> clusters;

	openImages  (inputDirPath,     data);
//	dbscanMain  (        data, clusters);
	partitionMethod(data, clusters);

//	cerr << "Total number of entries " << numberOfItems << endl;
	cerr << "Number of clusters " << clusters.size() << endl;

	assesClusters(clusters);
	saveClusters(  outputPath, clusters);

	cerr << "Done. Output saved to file." << endl;

	return 0;
}
