#include "XFeatRKNN.h"
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UFile.h>
#include <rtabmap/utilite/UDirectory.h>
#include <rtabmap/utilite/UConversion.h>
#include <rtabmap/utilite/UTimer.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace rtabmap
{

XFeatRKNN::XFeatRKNN(const ParametersMap & parameters) :
		modelPath_(Parameters::defaultXFeatRKNNModelPath()),
		coreMask_(Parameters::defaultXFeatRKNNCoreMask()),
		topK_(Parameters::defaultXFeatRKNNTopK()),
		threshold_(Parameters::defaultXFeatRKNNThreshold()),
		ctx_(0),
		initialized_(false),
		inputH_(0),
		inputW_(0),
		featsIndex_(-1),
		keypointsIndex_(-1),
		heatmapIndex_(-1)
{
	this->parseParameters(parameters);
}

XFeatRKNN::~XFeatRKNN()
{
	release();
}

void XFeatRKNN::release()
{
	if(initialized_)
	{
		rknn_destroy(ctx_);
		ctx_ = 0;
		initialized_ = false;
	}
}

void XFeatRKNN::parseParameters(const ParametersMap & parameters)
{
	Feature2D::parseParameters(parameters);

	std::string previousPath = modelPath_;
	int previousCoreMask = coreMask_;
	Parameters::parse(parameters, Parameters::kXFeatRKNNModelPath(), modelPath_);
	Parameters::parse(parameters, Parameters::kXFeatRKNNCoreMask(), coreMask_);
	Parameters::parse(parameters, Parameters::kXFeatRKNNTopK(), topK_);
	Parameters::parse(parameters, Parameters::kXFeatRKNNThreshold(), threshold_);

	modelPath_ = uReplaceChar(modelPath_, '~', UDirectory::homeDir());

	if(initialized_ && (modelPath_.compare(previousPath) != 0 || coreMask_ != previousCoreMask))
	{
		release();
	}
}

void XFeatRKNN::init()
{
	UASSERT(!initialized_);

	if(!UFile::exists(modelPath_) || UFile::getExtension(modelPath_).compare("rknn") != 0)
	{
		UERROR("XFeatRKNN: model path is not valid: \"%s\"=\"%s\"",
				Parameters::kXFeatRKNNModelPath().c_str(), modelPath_.c_str());
		return;
	}

	FILE * f = fopen(modelPath_.c_str(), "rb");
	UASSERT(f != 0);
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	std::vector<unsigned char> model(size);
	size_t read = fread(model.data(), 1, size, f);
	fclose(f);
	UASSERT((long)read == size);

	int ret = rknn_init(&ctx_, model.data(), size, 0, NULL);
	if(ret != RKNN_SUCC)
	{
		UERROR("XFeatRKNN: rknn_init failed (%d) for model \"%s\"", ret, modelPath_.c_str());
		return;
	}

	if(coreMask_ != 0)
	{
		ret = rknn_set_core_mask(ctx_, (rknn_core_mask)coreMask_);
		if(ret != RKNN_SUCC)
		{
			UWARN("XFeatRKNN: rknn_set_core_mask(%d) failed (%d), using auto.", coreMask_, ret);
		}
	}

	rknn_input_output_num ioNum;
	ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum));
	UASSERT(ret == RKNN_SUCC);
	UASSERT_MSG(ioNum.n_input == 1 && ioNum.n_output == 3,
			uFormat("XFeatRKNN: expected 1 input / 3 outputs, got %d/%d", ioNum.n_input, ioNum.n_output).c_str());

	rknn_tensor_attr inAttr;
	memset(&inAttr, 0, sizeof(inAttr));
	inAttr.index = 0;
	ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &inAttr, sizeof(inAttr));
	UASSERT(ret == RKNN_SUCC);
	// model exported as NCHW 1x1xHxW
	if(inAttr.fmt == RKNN_TENSOR_NCHW)
	{
		inputH_ = inAttr.dims[2];
		inputW_ = inAttr.dims[3];
	}
	else // native layout NHWC 1xHxWx1
	{
		inputH_ = inAttr.dims[1];
		inputW_ = inAttr.dims[2];
	}
	UASSERT(inputH_ > 0 && inputW_ > 0 && inputH_ % 8 == 0 && inputW_ % 8 == 0);

	// identify outputs by element count (independent of export order)
	int hc = inputH_ / 8;
	int wc = inputW_ / 8;
	for(unsigned int i = 0; i < ioNum.n_output; ++i)
	{
		rknn_tensor_attr attr;
		memset(&attr, 0, sizeof(attr));
		attr.index = i;
		ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
		UASSERT(ret == RKNN_SUCC);
		if((int)attr.n_elems == 64 * hc * wc)      featsIndex_ = i;
		else if((int)attr.n_elems == 65 * hc * wc) keypointsIndex_ = i;
		else if((int)attr.n_elems == hc * wc)      heatmapIndex_ = i;
	}
	UASSERT_MSG(featsIndex_ >= 0 && keypointsIndex_ >= 0 && heatmapIndex_ >= 0,
			"XFeatRKNN: could not identify feats/keypoints/heatmap outputs from tensor sizes");

	initialized_ = true;
	UINFO("XFeatRKNN: initialized, model=%s input=%dx%d coreMask=%d topK=%d threshold=%f",
			modelPath_.c_str(), inputW_, inputH_, coreMask_, topK_, threshold_);
}

std::vector<cv::KeyPoint> XFeatRKNN::generateKeypointsImpl(const cv::Mat & image, const cv::Rect & roi, const cv::Mat & mask)
{
	UASSERT(!image.empty() && image.channels() == 1 && image.depth() == CV_8U);
	descriptors_ = cv::Mat();
	std::vector<cv::KeyPoint> keypoints;

	if(!initialized_)
	{
		init();
		if(!initialized_)
		{
			return keypoints;
		}
	}

	UTimer timer;
	cv::Mat imgRoi(image, roi);
	int h0 = imgRoi.rows;
	int w0 = imgRoi.cols;

	// two-step resize: integer-ratio INTER_AREA fast path, then small linear squeeze
	cv::Mat resized = imgRoi;
	if(w0 != inputW_ || h0 != inputH_)
	{
		if(w0 % inputW_ == 0)
		{
			cv::resize(imgRoi, resized, cv::Size(inputW_, h0 * inputW_ / w0), 0, 0, cv::INTER_AREA);
		}
		if(resized.cols != inputW_ || resized.rows != inputH_)
		{
			cv::resize(resized, resized, cv::Size(inputW_, inputH_), 0, 0, cv::INTER_LINEAR);
		}
	}

	// per-image standardization == the removed InstanceNorm (affine-invariant,
	// so mean/std computed on uint8 without /255 gives the identical output)
	cv::Scalar mean, stddev;
	cv::meanStdDev(resized, mean, stddev);
	cv::Mat input;
	resized.convertTo(input, CV_32F, 1.0 / (stddev[0] + 1e-5), -mean[0] / (stddev[0] + 1e-5));
	UASSERT(input.isContinuous());

	rknn_input in;
	memset(&in, 0, sizeof(in));
	in.index = 0;
	in.type = RKNN_TENSOR_FLOAT32;
	// single-channel: NHWC 1xHxWx1 is byte-identical to NCHW 1x1xHxW, and the
	// runtime's native input layout is NHWC (NCHW is rejected by rknn_inputs_set)
	in.fmt = RKNN_TENSOR_NHWC;
	in.size = inputH_ * inputW_ * sizeof(float);
	in.buf = input.data;
	int ret = rknn_inputs_set(ctx_, 1, &in);
	UASSERT(ret == RKNN_SUCC);

	ret = rknn_run(ctx_, NULL);
	UASSERT(ret == RKNN_SUCC);

	rknn_output outputs[3];
	memset(outputs, 0, sizeof(outputs));
	for(int i = 0; i < 3; ++i)
	{
		outputs[i].index = i;
		outputs[i].want_float = 1;
	}
	ret = rknn_outputs_get(ctx_, 3, outputs, NULL);
	UASSERT(ret == RKNN_SUCC);

	const float * feats = (const float *)outputs[featsIndex_].buf;      // 64 x hc x wc
	const float * klogits = (const float *)outputs[keypointsIndex_].buf; // 65 x hc x wc
	const float * heat = (const float *)outputs[heatmapIndex_].buf;      // hc x wc

	int hc = inputH_ / 8;
	int wc = inputW_ / 8;
	int cells = hc * wc;

	// softmax over 65 channels per cell, drop dustbin, pixel-shuffle to full-res score map
	cv::Mat scores(inputH_, inputW_, CV_32F);
	for(int cell = 0; cell < cells; ++cell)
	{
		float maxLogit = klogits[cell];
		for(int c = 1; c < 65; ++c)
		{
			float v = klogits[c * cells + cell];
			if(v > maxLogit) maxLogit = v;
		}
		float sum = 0.0f;
		float e[64];
		for(int c = 0; c < 64; ++c)
		{
			e[c] = std::exp(klogits[c * cells + cell] - maxLogit);
			sum += e[c];
		}
		sum += std::exp(klogits[64 * cells + cell] - maxLogit); // dustbin
		float inv = 1.0f / sum;
		int ci = cell / wc;
		int cj = cell % wc;
		for(int c = 0; c < 64; ++c)
		{
			scores.at<float>(ci * 8 + c / 8, cj * 8 + c % 8) = e[c] * inv;
		}
	}

	// 5x5 local-max NMS + threshold, weighted by cell reliability
	cv::Mat localMax;
	cv::dilate(scores, localMax, cv::Mat::ones(5, 5, CV_32F));

	struct Cand { float x, y, s; };
	std::vector<Cand> cands;
	cands.reserve(2048);
	float scaleX = (float)w0 / (float)inputW_;
	float scaleY = (float)h0 / (float)inputH_;
	for(int y = 0; y < inputH_; ++y)
	{
		const float * sRow = scores.ptr<float>(y);
		const float * mRow = localMax.ptr<float>(y);
		for(int x = 0; x < inputW_; ++x)
		{
			if(sRow[x] > threshold_ && sRow[x] >= mRow[x])
			{
				float s = sRow[x] * heat[(y / 8) * wc + (x / 8)];
				// mask is in full image coordinates
				int fullX = (int)(x * scaleX) + roi.x;
				int fullY = (int)(y * scaleY) + roi.y;
				if(mask.empty() ||
				   (fullX >= 0 && fullX < mask.cols && fullY >= 0 && fullY < mask.rows &&
				    mask.at<unsigned char>(fullY, fullX) != 0))
				{
					cands.push_back({(float)x, (float)y, s});
				}
			}
		}
	}

	if((int)cands.size() > topK_)
	{
		std::nth_element(cands.begin(), cands.begin() + topK_, cands.end(),
				[](const Cand & a, const Cand & b) {return a.s > b.s;});
		cands.resize(topK_);
	}

	// bilinear-sample 64-D descriptors at cell coordinates, L2-normalize
	keypoints.reserve(cands.size());
	descriptors_ = cv::Mat((int)cands.size(), 64, CV_32F);
	for(size_t i = 0; i < cands.size(); ++i)
	{
		float fx = cands[i].x / 8.0f - 0.5f;
		float fy = cands[i].y / 8.0f - 0.5f;
		fx = std::min(std::max(fx, 0.0f), wc - 1.001f);
		fy = std::min(std::max(fy, 0.0f), hc - 1.001f);
		int x0 = (int)fx;
		int y0 = (int)fy;
		int x1 = std::min(x0 + 1, wc - 1);
		int y1 = std::min(y0 + 1, hc - 1);
		float ax = fx - x0;
		float ay = fy - y0;

		float * d = descriptors_.ptr<float>((int)i);
		float norm = 0.0f;
		for(int c = 0; c < 64; ++c)
		{
			const float * fc = feats + c * cells;
			float v = fc[y0 * wc + x0] * (1.0f - ax) * (1.0f - ay)
					+ fc[y0 * wc + x1] * ax * (1.0f - ay)
					+ fc[y1 * wc + x0] * (1.0f - ax) * ay
					+ fc[y1 * wc + x1] * ax * ay;
			d[c] = v;
			norm += v * v;
		}
		norm = 1.0f / (std::sqrt(norm) + 1e-8f);
		for(int c = 0; c < 64; ++c)
		{
			d[c] *= norm;
		}

		keypoints.push_back(cv::KeyPoint(cands[i].x * scaleX, cands[i].y * scaleY, 8, -1, cands[i].s));
	}

	rknn_outputs_release(ctx_, 3, outputs);

	UDEBUG("XFeatRKNN: %d keypoints in %fs", (int)keypoints.size(), timer.ticks());

	// Apply limitKeypoints to enforce maxFeatures and SSC
	this->limitKeypoints(keypoints, descriptors_, this->getMaxFeatures(), cv::Size(roi.width, roi.height), this->getSSC());

	return keypoints;
}

cv::Mat XFeatRKNN::generateDescriptorsImpl(const cv::Mat & image, std::vector<cv::KeyPoint> & keypoints) const
{
	if(!keypoints.empty() && (int)keypoints.size() != descriptors_.rows)
	{
		UERROR("XFeatRKNN: the number of keypoints (%ld) doesn't match the number of buffered "
			"descriptors (%d). Descriptors extraction should be called right after keypoints "
			"detection with the same keypoints. Returning empty descriptors.",
			keypoints.size(), descriptors_.rows);
		return cv::Mat();
	}
	return descriptors_;
}

}
