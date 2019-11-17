#define  LBP_INPUT_SIZE 16
#define  LBP_DIMEN_SIZE 52

cv::Mat lbpCut(cv::Mat origImg, int x, int y);
cv::Mat lbpImg(cv::Mat origImg);

int lbpComp(cv::Mat ref, cv::Mat tar, int dimension);

int savelbp(cv::Mat lbpImg, std::string lbpFile);
cv::Mat loadlbp(std::string lbpFile);