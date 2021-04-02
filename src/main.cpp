extern "C"
{
    #include "lsd.h"
};
#include "VPDetection.h"

using namespace std;
using namespace cv;

void printHelp(){
    cout << endl;
    cout << "Generate Image with vanishing point." << endl;
    cout << "Usage:" << endl;
    cout << "\tVanishingPoint -s=ImageFilePath [-o=ImageOutputName]" << endl;
    cout << "Options:" << endl;
    cout << "\t -s=<path> \t indicate the location of the input image." << endl;
    cout << "\t -o=<path> \t indicate the location of the output image. If absent, only printed." << endl;
    cout << "\t -vp \t\t if set, store the vanishing point info to json file" << endl;
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

	// 需要画出的灭点线段数量
    int drawVPNum = clusters.size();  // 3
    for ( int i = 0; i < drawVPNum; ++i )
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

/**
 * 生成的 ***_vps.json 文件在可执行文件的同一目录
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv)
{
    const string keys =
            "{help h |      | print this message   }"
            "{s | | the input image, required}"
            "{o | false | Optional, the output image path, if absent, only printed}"
            "{vp| false | Optional, dump the vanishing point to json file}";
    cv::CommandLineParser parser(argc, argv, keys);

    string inputImage = parser.get<string>("s");
    if (parser.has("help") || inputImage.empty() || inputImage == "true") {
        printHelp();
        return 0;
    }

	cv::Mat image= cv::imread(inputImage );
	if ( image.empty() )
	{
		printf("Load image error : %s\n", inputImage.c_str() );
		return -1;
	}
	bool dump2json = parser.get<bool>("vp");

	// LSD line segment detection
	double thLength = 30.0;
	std::vector<std::vector<double> > lines;
	LineDetect( image, thLength, lines );

	// Camera internal parameters
//	cv::Point2d pp( 307, 251 );        // Principle point (in pixel)
//	double f = 6.053 / 0.009;          // Focal length (in pixel)
    cv::Point2d pp(image.cols / 2, image.rows / 2);
    double f = 1.2*(std::max(image.cols, image.rows));

	// Vanishing point detection
	std::vector<cv::Point3d> vps;              // Detected vanishing points (in pixel)
	std::vector<std::vector<int> > clusters;   // Line segment clustering results of each vanishing point
	VPDetection detector(dump2json);
    // 设置输出的灭点文件名
	string vpJsonFile = inputImage.substr(inputImage.find_last_of('/')+1);
	vpJsonFile = vpJsonFile.substr(0, vpJsonFile.find('.'));
	detector.setVpJsonFile(vpJsonFile);

	detector.run( lines, pp, f, vps, clusters );

	string outputImg = parser.get<string>("o");
    if (outputImg.length() > 0 && outputImg != "false" && outputImg != "true") {

        drawClusters( image, lines, clusters );
        imshow("",image);
        cv::waitKey( 0 );

        cv::imwrite(outputImg, image);
        cout << "Output image saved." << endl;
    }
    return 0;
}
