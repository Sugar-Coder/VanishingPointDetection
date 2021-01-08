extern "C"
{
    #include "lsd.h"
};
#include "VPDetection.h"

using namespace std;
using namespace cv;

struct ArgInfo{
    std::string inputImg;
    bool doSave;
    std::string outputImg;

    explicit ArgInfo(std::string inFile) : inputImg(inFile), doSave(false) {}
    ArgInfo(std::string inFile, std::string outFile) : inputImg(inFile), doSave(true), outputImg(outFile) {}
};

void printHelp(){
    cout << endl;
    cout << "Generate Image with vanishing point." << endl;
    cout << "Usage:" << endl;
    cout << "\tVanishingPoint -s ImageFilePath [-o ImageOutputName]" << endl;
    cout << "Options:" << endl;
    cout << "\t -s <path> \t indicate the location of the input image." << endl;
    cout << "\t -o <path> \t indicate the location of the output image. If absent, only printed." << endl;
}

ArgInfo parseArg(int argc, char** argv) {
    if (argc == 3 && !strcmp("-s", argv[1])) {
        return ArgInfo(argv[2]);
    }

    if (argc == 5) {
        if (!strcmp("-s", argv[1]) && !strcmp("-o", argv[3])) {
            return ArgInfo(argv[2], argv[4]);
        }
        if (!strcmp("-s", argv[3]) && !strcmp("-o", argv[1])) {
            return ArgInfo(argv[4], argv[1]);
        }
    }

    printHelp();
    exit(0);
}


// LSD line segment detection
void LineDetect( cv::Mat image, double thLength, std::vector<std::vector<double> > &lines )
{
	cv::Mat grayImage;
	if ( image.channels() == 1 )
		grayImage = image;
	else
		cv::cvtColor(image, grayImage, COLOR_BGR2GRAY);

	image_double imageLSD = new_image_double( grayImage.cols, grayImage.rows );
	unsigned char* im_src = (unsigned char*) grayImage.data;

	int xsize = grayImage.cols;
	int ysize = grayImage.rows;
	for ( int y = 0; y < ysize; ++y )
	{
		for ( int x = 0; x < xsize; ++x )
		{
			imageLSD->data[y * xsize + x] = im_src[y * xsize + x];
		}
	}

	ntuple_list linesLSD = lsd( imageLSD );
	free_image_double( imageLSD );

	int nLines = linesLSD->size;
	int dim = linesLSD->dim;
	std::vector<double> lineTemp( 4 );
	for ( int i = 0; i < nLines; ++i )
	{
		double x1 = linesLSD->values[i * dim + 0];
		double y1 = linesLSD->values[i * dim + 1];
		double x2 = linesLSD->values[i * dim + 2];
		double y2 = linesLSD->values[i * dim + 3];

		double l = sqrt( ( x1 - x2 ) * ( x1 - x2 ) + ( y1 - y2 ) * ( y1 - y2 ) );
		if ( l > thLength )
		{
			lineTemp[0] = x1;
			lineTemp[1] = y1;
			lineTemp[2] = x2;
			lineTemp[3] = y2;

			lines.push_back( lineTemp );
		}
	}

	free_ntuple_list(linesLSD);
}

void drawClusters( cv::Mat &img, std::vector<std::vector<double> > &lines, std::vector<std::vector<int> > &clusters )
{
	int cols = img.cols;
	int rows = img.rows;

	//draw lines
	std::vector<cv::Scalar> lineColors( 3 );
	lineColors[0] = cv::Scalar( 0, 0, 255 );
	lineColors[1] = cv::Scalar( 0, 255, 0 );
	lineColors[2] = cv::Scalar( 255, 0, 0 );

	for ( int i=0; i<lines.size(); ++i )
	{
		int idx = i;
		cv::Point pt_s = cv::Point( lines[idx][0], lines[idx][1]);
		cv::Point pt_e = cv::Point( lines[idx][2], lines[idx][3]);
		cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

		cv::line( img, pt_s, pt_e, cv::Scalar(0,0,0), 2, LINE_AA);
	}

	for ( int i = 0; i < clusters.size(); ++i )
	{
		for ( int j = 0; j < clusters[i].size(); ++j )
		{
			int idx = clusters[i][j];

			cv::Point pt_s = cv::Point( lines[idx][0], lines[idx][1] );
			cv::Point pt_e = cv::Point( lines[idx][2], lines[idx][3] );
			cv::Point pt_m = ( pt_s + pt_e ) * 0.5;

			cv::line( img, pt_s, pt_e, lineColors[i], 2, LINE_AA );
		}
	}
}

int main(int argc, char** argv)
{
    ArgInfo argInfo = parseArg(argc, argv);
	string inPutImage = argInfo.inputImg;

	cv::Mat image= cv::imread( inPutImage );
	if ( image.empty() )
	{
		printf( "Load image error : %s\n", inPutImage.c_str() );
		return -1;
	}

	// LSD line segment detection
	double thLength = 30.0;
	std::vector<std::vector<double> > lines;
	LineDetect( image, thLength, lines );

	// Camera internal parameters
	cv::Point2d pp( 307, 251 );        // Principle point (in pixel)
	double f = 6.053 / 0.009;          // Focal length (in pixel)

	// Vanishing point detection
	std::vector<cv::Point3d> vps;              // Detected vanishing points (in pixel)
	std::vector<std::vector<int> > clusters;   // Line segment clustering results of each vanishing point
	VPDetection detector;
	detector.run( lines, pp, f, vps, clusters );

	drawClusters( image, lines, clusters );
	imshow("",image);
	cv::waitKey( 0 );
    if (argInfo.doSave) {
        cv::imwrite(argInfo.outputImg, image);
        cout << "Output image saved." << endl;
    }
}
