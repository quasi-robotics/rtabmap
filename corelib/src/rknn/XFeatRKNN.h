/**
 * XFeat local feature detector running on a Rockchip NPU via the RKNN C API.
 *
 * Expects an XFeat backbone exported with a static input (1x1xHxW, grayscale)
 * and the input InstanceNorm removed from the graph (per-image standardization
 * is done here on CPU), producing the dense outputs:
 *   feats     1 x 64 x H/8 x W/8   (descriptor map)
 *   keypoints 1 x 65 x H/8 x W/8   (in-cell logits, channel 64 = dustbin)
 *   heatmap   1 x  1 x H/8 x W/8   (reliability)
 * Top-k selection, NMS and descriptor sampling are done on CPU.
 */

#ifndef XFEATRKNN_H
#define XFEATRKNN_H

#include <rtabmap/core/Features2d.h>
#include <opencv2/core/mat.hpp>
#include <vector>

#include <rknn/rknn_api.h>

namespace rtabmap
{

class XFeatRKNN : public Feature2D
{
public:
	XFeatRKNN(const ParametersMap & parameters = ParametersMap());
	virtual ~XFeatRKNN();

	virtual void parseParameters(const ParametersMap & parameters);
	virtual Feature2D::Type getType() const {return kFeatureXFeatRKNN;}

private:
	virtual std::vector<cv::KeyPoint> generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi, const cv::Mat & mask = cv::Mat());
	virtual cv::Mat generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const;

	void init();
	void release();

private:
	std::string modelPath_;
	int coreMask_;
	int topK_;
	float threshold_;

	rknn_context ctx_;
	bool initialized_;
	int inputH_;
	int inputW_;
	int featsIndex_;
	int keypointsIndex_;
	int heatmapIndex_;

	cv::Mat descriptors_;
};

}

#endif
