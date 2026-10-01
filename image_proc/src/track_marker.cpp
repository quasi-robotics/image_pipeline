// Copyright 2024 Open Navigation LLC
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the copyright holder nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <image_proc/track_marker.hpp>
#include <image_proc/utils.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/calib3d.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Matrix3x3.hpp>

namespace image_proc
{

TrackMarkerNode::TrackMarkerNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("TrackMarkerNode", options)
{
  // TransportHints does not actually declare the parameter
  this->declare_parameter<std::string>("image_transport", "raw");

  // For compressed topics to remap appropriately, we need to pass a
  // fully expanded and remapped topic name to image_transport
  auto node_base = this->get_node_base_interface();
  image_topic_ = node_base->resolve_topic_or_service_name("image", false);

  // Declare parameters before we setup any publishers or subscribers
  marker_id_ = this->declare_parameter("marker_id", 0);
  marker_size_ = this->declare_parameter("marker_size", 0.05);
  // Default dictionary is cv::aruco::DICT_6X6_250
  int dict_id = this->declare_parameter("dictionary", 10);
  ambiguity_ratio_ = this->declare_parameter("ambiguity_ratio", 0.0);
  vertical_frame_ = this->declare_parameter("vertical_frame", std::string(""));
  expected_tilt_ = this->declare_parameter("expected_tilt", 0.0);
  vertical_tilt_margin_ = this->declare_parameter("vertical_tilt_margin", 0.05);
  if (!vertical_frame_.empty()) {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  }
  const bool refine_corners = this->declare_parameter("refine_corners", false);

  #if CV_VERSION_MAJOR > 4 || CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7
  detector_params_ = cv::makePtr<cv::aruco::DetectorParameters>();
  #else
  detector_params_ = cv::aruco::DetectorParameters::create();
  #endif
  if (refine_corners) {
    detector_params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
  }

  #if CV_VERSION_MAJOR > 4 || CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7
  dictionary_ = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(dict_id));
  #else
  dictionary_ = cv::aruco::getPredefinedDictionary(dict_id);
  #endif

  // Setup lazy subscriber using publisher connection callback
  rclcpp::PublisherOptions pub_options;
  pub_options.event_callbacks.matched_callback =
    [this](rclcpp::MatchedInfo &)
    {
      if (pub_->get_subscription_count() == 0) {
        sub_camera_.shutdown();
      } else if (!sub_camera_) {
        // Create subscriber with QoS matched to subscribed topic publisher
        auto qos_profile = getTopicQosProfile(this, image_topic_);
        image_transport::TransportHints hints(this);
        sub_camera_ = image_transport::create_camera_subscription(
          this, image_topic_, std::bind(
            &TrackMarkerNode::imageCb,
            this, std::placeholders::_1, std::placeholders::_2),
          hints.getTransport(), qos_profile);
      }
    };

  // Create publisher - allow overriding QoS settings (history, depth, reliability)
  pub_options.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
  pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "tracked_pose", 10, pub_options);

  on_set_parameters_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&TrackMarkerNode::paramCallback, this, std::placeholders::_1));
}

rcl_interfaces::msg::SetParametersResult TrackMarkerNode::paramCallback(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto & parameter : parameters) {
    if (parameter.get_name() == "marker_id") {
      const int64_t marker_id = parameter.as_int();
      if (marker_id < 0) {
        result.successful = false;
        result.reason = "marker_id must be >= 0";
        return result;
      }
      marker_id_ = static_cast<int>(marker_id);
      RCLCPP_INFO(get_logger(), "Tracking marker id %d", marker_id_.load());
    }
  }
  return result;
}

void TrackMarkerNode::imageCb(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info_msg)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(image_msg);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners;
  cv::aruco::detectMarkers(
    cv_ptr->image, dictionary_, marker_corners, marker_ids, detector_params_);

  for (size_t i = 0; i < marker_ids.size(); ++i) {
    if (marker_ids[i] == marker_id_) {
      // This is our desired marker
      std::vector<std::vector<cv::Point2f>> corners;
      corners.push_back(marker_corners[i]);

      // Copy the matrices since they are const and OpenCV functions are not
      auto k = info_msg->k;
      auto d = info_msg->d;

      // Get the camera info
      cv::Mat intrinsics(3, 3, CV_64FC1, reinterpret_cast<void *>(k.data()));
      cv::Mat dist_coeffs(info_msg->d.size(), 1, CV_64FC1, reinterpret_cast<void *>(d.data()));

      // Estimate pose
      std::vector<cv::Vec3d> rvecs, tvecs;
      if (ambiguity_ratio_ > 0.0) {
        // A small planar marker has two poses (mirrored about the line of sight) that fit
        // its corners almost equally well; only publish when one is clearly better
        const float h = marker_size_ / 2.0;
        const std::vector<cv::Point3f> object_points{
          {-h, h, 0}, {h, h, 0}, {h, -h, 0}, {-h, -h, 0}};
        std::vector<cv::Mat> solution_rvecs, solution_tvecs;
        std::vector<double> errors;
        cv::solvePnPGeneric(
          object_points, marker_corners[i], intrinsics, dist_coeffs, solution_rvecs,
          solution_tvecs, false, cv::SOLVEPNP_IPPE_SQUARE, cv::noArray(), cv::noArray(), errors);
        if (errors.size() < 2) {
          continue;
        }
        int choice = 0;
        if (errors[1] < ambiguity_ratio_ * errors[0]) {
          choice = vertical_frame_.empty() ? -1 :
            chooseTiltSolution(solution_rvecs, image_msg->header);
          if (choice < 0) {
            RCLCPP_DEBUG(
              this->get_logger(), "Ambiguous marker pose (reprojection errors %.3f, %.3f px)",
              errors[0], errors[1]);
            continue;
          }
        }
        rvecs.emplace_back(solution_rvecs[choice]);
        tvecs.emplace_back(solution_tvecs[choice]);
      } else {
        cv::aruco::estimatePoseSingleMarkers(
          corners, marker_size_, intrinsics, dist_coeffs, rvecs, tvecs);
      }

      // Publish pose of marker
      geometry_msgs::msg::PoseStamped pose;
      pose.header = image_msg->header;
      // Fill in pose
      pose.pose.position.x = tvecs[0][0];
      pose.pose.position.y = tvecs[0][1];
      pose.pose.position.z = tvecs[0][2];
      // Convert angle-axis to quaternion
      const tf2::Vector3 rvec(rvecs[0][0], rvecs[0][1], rvecs[0][2]);
      const tf2::Quaternion q(rvec.normalized(), rvec.length());
      tf2::convert(q, pose.pose.orientation);
      pub_->publish(pose);
    }
  }
}

int TrackMarkerNode::chooseTiltSolution(
  const std::vector<cv::Mat> & rvecs, const std_msgs::msg::Header & header)
{
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_->lookupTransform(vertical_frame_, header.frame_id, tf2::TimePointZero);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "Can't check marker tilt: %s",
      ex.what());
    return -1;
  }
  tf2::Quaternion q;
  tf2::fromMsg(transform.transform.rotation, q);
  const tf2::Matrix3x3 camera_to_frame(q);

  // How far each solution's marker normal (its z axis) tilts from the expected tilt
  double tilt[2];
  for (int i = 0; i < 2; ++i) {
    cv::Mat rotation;
    cv::Rodrigues(rvecs[i], rotation);
    const tf2::Vector3 normal = camera_to_frame * tf2::Vector3(
      rotation.at<double>(0, 2), rotation.at<double>(1, 2), rotation.at<double>(2, 2));
    tilt[i] = std::fabs(std::asin(std::clamp(normal.z(), -1.0, 1.0)) - expected_tilt_);
  }
  if (std::fabs(tilt[0] - tilt[1]) < vertical_tilt_margin_) {
    return -1;
  }
  return tilt[0] <= tilt[1] ? 0 : 1;
}

}  // namespace image_proc

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(image_proc::TrackMarkerNode)
