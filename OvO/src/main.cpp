#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <gazebo_msgs/srv/spawn_entity.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>  // Added for waypoint (clicked point)
#include <opencv2/opencv.hpp>
#include "std_msgs/msg/float64.hpp"
#include <nav_msgs/msg/occupancy_grid.hpp>
#include "nav_msgs/msg/odometry.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <cmath>
#include <vector>
#include <optional>
#include <chrono>
#include <cmath>

using namespace std::chrono_literals;

class System : public rclcpp::Node{
public:
    using NavigateToPose = nav2_msgs::action::NavigateToPose;
    using GoalHandleNavigateToPose = rclcpp_action::ClientGoalHandle<NavigateToPose>;
    System() : Node("system_node"){
        publish_timer_ = this->create_wall_timer(10ms, std::bind(&System::system, this));
        laserscan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&System::laser_callback, this, std::placeholders::_1));
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&System::odomCallback, this, std::placeholders::_1));
        action_client_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");
        
        // tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        // tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        // nav2_client_ = rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(this, "navigate_to_pose");

        this->declare_parameter<int>("Img Size", 1000);
        this->declare_parameter<double>("Lidar Min", 0.2);
        this->declare_parameter<double>("Lidar Max", 1.50);
        this->declare_parameter<double>("Pursuit Object Size", 0.1);
        this->declare_parameter<double>("Filtered Radius", 18);
        this->declare_parameter<double>("Density Radius", 5.0);
        this->declare_parameter<int>("Density Thresh", 3);
        this->declare_parameter<double>("Temp", 25.0);
        this->declare_parameter<double>("Temp2", 5.0);
        this->declare_parameter<double>("Temp3", 1.0);

    }

    void laser_callback(const std::shared_ptr<sensor_msgs::msg::LaserScan> msg){latest_laserscan_ = msg;}

    void odomCallback(const std::shared_ptr<nav_msgs::msg::Odometry> msg){latest_odom_ = msg;}

    cv::Mat laserScanToMat(const sensor_msgs::msg::LaserScan::SharedPtr& scan, int img_size, float rotation) {
        // Parameters
        float max_range = scan->range_max;
        cv::Mat image = cv::Mat::zeros(img_size, img_size, CV_8UC1);

        for (size_t i = 0; i < scan->ranges.size(); i++) {
            float range = scan->ranges[i];
            if (range > scan->range_min && range < scan->range_max) {
                float angle = (scan->angle_min + i * scan->angle_increment) + rotation;
                int x = static_cast<int>((range * cos(angle)) * img_size / (2 * max_range)) + img_size / 2;
                int y = static_cast<int>((range * sin(angle)) * img_size / (2 * max_range)) + img_size / 2;
                if (x >= 0 && x < img_size && y >= 0 && y < img_size) {
                    image.at<uchar>(y, x) = 255;
                }
            }
        }

        return image;
    }

    cv::Mat laserScanToMat(const sensor_msgs::msg::LaserScan::SharedPtr& scan, int img_size, float rotation, double min, double max) {
        // Parameters
        float max_range = scan->range_max;
        cv::Mat image = cv::Mat::zeros(img_size, img_size, CV_8UC1);

        for (size_t i = 0; i < scan->ranges.size(); i++) {
            float range = scan->ranges[i];

            // Check if the range is within the sensor's range and the specified min-max range
            if (range > scan->range_min && range < scan->range_max && range >= min && range <= max) {
                float angle = (scan->angle_min + i * scan->angle_increment) + rotation;
                int x = static_cast<int>((range * cos(angle)) * img_size / (2 * max_range)) + img_size / 2;
                int y = static_cast<int>((range * sin(angle)) * img_size / (2 * max_range)) + img_size / 2;

                // Check if (x, y) coordinates are within image bounds
                if (x >= 0 && x < img_size && y >= 0 && y < img_size) {
                    image.at<uchar>(y, x) = 255;
                }
            }
        }

        return image;
    }

    std::vector<cv::Point> laserScanToMatVector(const sensor_msgs::msg::LaserScan::SharedPtr& scan, int img_size, float rotation, double min, double max) {
        // Parameters
        float max_range = scan->range_max;
        std::vector<cv::Point> valid_points;  // Vector to store valid points

        for (size_t i = 0; i < scan->ranges.size(); i++) {
            float range = scan->ranges[i];

            // Check if the range is within the sensor's range and the specified min-max range
            if (range > scan->range_min && range < scan->range_max && range >= min && range <= max) {
                float angle = (scan->angle_min + i * scan->angle_increment) + rotation;
                int x = static_cast<int>((range * cos(angle)) * img_size / (2 * max_range)) + img_size / 2;
                int y = static_cast<int>((range * sin(angle)) * img_size / (2 * max_range)) + img_size / 2;

                // Check if (x, y) coordinates are within image bounds
                if (x >= 0 && x < img_size && y >= 0 && y < img_size) {
                    valid_points.emplace_back(x, y);  // Store the valid point
                }
            }
        }

        return valid_points;  // Return the vector of valid points
    }

    // Function to replace one color with another in a cv::Mat image
    void replaceColor(cv::Mat& image, const cv::Scalar& target_color, const cv::Scalar& replacement_color, int tolerance = 0) {
        // Convert the image to a format that allows direct access to each pixel if necessary
        if (image.channels() != 3) {
            throw std::invalid_argument("replaceColor function requires a 3-channel color image.");
        }

        // Iterate over every pixel in the image
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                // Get the pixel color as a Vec3b (BGR)
                cv::Vec3b& pixel = image.at<cv::Vec3b>(y, x);

                // Check if the pixel color matches the target color within the given tolerance
                if (std::abs(pixel[0] - target_color[0]) <= tolerance &&
                    std::abs(pixel[1] - target_color[1]) <= tolerance &&
                    std::abs(pixel[2] - target_color[2]) <= tolerance) {
                    
                    // Replace the color with the replacement color
                    pixel[0] = static_cast<uchar>(replacement_color[0]);
                    pixel[1] = static_cast<uchar>(replacement_color[1]);
                    pixel[2] = static_cast<uchar>(replacement_color[2]);
                }
            }
        }
    }

    // Image Processing: Blend 2 images over each other (Blend mode: Screen) // Note that this is Exact Image Sizes
    void blendImages(const cv::Mat& baseImage, const cv::Mat& overlayImage, cv::Mat& outputImage){
        cv::Mat overlayImageResized;
        cv::resize(overlayImage, overlayImageResized, baseImage.size());
        cv::Mat baseImageConverted, overlayImageConverted;

        if(baseImage.channels() == 3){baseImageConverted = baseImage;}
            else if(baseImage.channels() == 1){cv::cvtColor(baseImage, baseImageConverted, cv::COLOR_GRAY2RGB);}
            else{cv::cvtColor(baseImage, baseImageConverted, cv::COLOR_RGBA2RGB);
        }

        if(overlayImageResized.channels() == 3){overlayImageConverted = overlayImageResized;}
            else if(overlayImageResized.channels() == 1){cv::cvtColor(overlayImageResized, overlayImageConverted, cv::COLOR_GRAY2RGB);}
            else{cv::cvtColor(overlayImageResized, overlayImageConverted, cv::COLOR_RGBA2RGB);
        }

        cv::Mat mask;
        cv::inRange(overlayImageConverted, cv::Scalar(0, 0, 0), cv::Scalar(10, 10, 10), mask);

        cv::Mat invMask;
        cv::bitwise_not(mask, invMask);

        outputImage = baseImageConverted.clone();

        cv::Mat foreground, background;
        baseImageConverted.copyTo(background, mask);
        overlayImageConverted.copyTo(foreground, invMask);

        cv::add(background, foreground, outputImage);
    }

    cv::Mat convertChannels(const cv::Mat& source, const cv::Mat& target) {
        cv::Mat converted;

        // Check if the number of channels is different
        if (source.channels() != target.channels()) {
            if (target.channels() == 3 && source.channels() == 1) {
                // Convert grayscale to BGR
                cv::cvtColor(source, converted, cv::COLOR_GRAY2BGR);
            } else if (target.channels() == 1 && source.channels() == 3) {
                // Convert BGR to grayscale
                cv::cvtColor(source, converted, cv::COLOR_BGR2GRAY);
            } else if (target.channels() == 4 && source.channels() == 3) {
                // Convert BGR to BGRA
                cv::cvtColor(source, converted, cv::COLOR_BGR2BGRA);
            } else if (target.channels() == 3 && source.channels() == 4) {
                // Convert BGRA to BGR
                cv::cvtColor(source, converted, cv::COLOR_BGRA2BGR);
            } else {
                // Other conversions can be added as needed
                throw std::invalid_argument("Unsupported channel conversion.");
            }
        } else {
            // No conversion needed, return the original
            converted = source.clone();
        }

        return converted;
    }

    // Function to convert cv::Mat image to a different number of channels
    cv::Mat convertChannels(const cv::Mat& image, int desired_channels) {
        cv::Mat converted_image;

        if (image.channels() == desired_channels) {
            // If the image already has the desired number of channels, return a copy
            converted_image = image.clone();
        } else if (image.channels() == 1 && desired_channels == 3) {
            // Convert single-channel (grayscale) image to 3-channel (BGR) image
            cv::cvtColor(image, converted_image, cv::COLOR_GRAY2BGR);
        } else if (image.channels() == 3 && desired_channels == 1) {
            // Convert 3-channel (BGR) image to single-channel (grayscale) image
            cv::cvtColor(image, converted_image, cv::COLOR_BGR2GRAY);
        } else if (image.channels() == 1 && desired_channels == 4) {
            // Convert single-channel (grayscale) image to 4-channel (BGRA) image
            cv::cvtColor(image, converted_image, cv::COLOR_GRAY2BGRA);
        } else if (image.channels() == 3 && desired_channels == 4) {
            // Convert 3-channel (BGR) image to 4-channel (BGRA) image
            cv::cvtColor(image, converted_image, cv::COLOR_BGR2BGRA);
        } else if (image.channels() == 4 && desired_channels == 3) {
            // Convert 4-channel (BGRA) image to 3-channel (BGR) image
            cv::cvtColor(image, converted_image, cv::COLOR_BGRA2BGR);
        } else if (image.channels() == 4 && desired_channels == 1) {
            // Convert 4-channel (BGRA) image to 1-channel (grayscale) image
            cv::cvtColor(image, converted_image, cv::COLOR_BGRA2GRAY);
        } else {
            throw std::invalid_argument("Unsupported channel conversion.");
        }

        return converted_image;
    }

    void findCircle(double x1, double y1, double x2, double y2, double x3, double y3, double& x_c, double& y_c, double& radius) {
        // Calculate the determinants
        double A = x1 * x1 + y1 * y1;
        double B = x2 * x2 + y2 * y2;
        double C = x3 * x3 + y3 * y3;

        // Calculate the determinant D
        double D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
        if (D == 0) {
            throw std::invalid_argument("The points are collinear.");
        }

        // Calculate the center coordinates
        x_c = (A * (y2 - y3) + B * (y3 - y1) + C * (y1 - y2)) / D;
        y_c = (A * (x3 - x2) + B * (x1 - x3) + C * (x2 - x1)) / D;

        // Calculate the radius
        radius = std::sqrt((x_c - x1) * (x_c - x1) + (y_c - y1) * (y_c - y1));
    }

    std::vector<cv::Scalar> findCircles(const std::vector<cv::Point>& points, int step) {
        std::vector<cv::Scalar> circles;  // Vector to store the circles

        // Ensure there are at least 3 points to form a circle
        if (points.size() < 3) {
            throw std::invalid_argument("At least three points are required to form a circle.");
        }

        size_t num_points = points.size();

        // Loop through every point
        for (size_t i = 0; i < num_points; ++i) {
            // For each point, iterate with step to get the other two points
            for (size_t j = 1; j <= 2; ++j) {
                size_t idx2 = (i + j * step) % num_points;   // Step through for the second point
                size_t idx3 = (i + (j + 1) * step) % num_points; // Step through for the third point

                // Get three points: current, the one at the step distance, and the next one after that
                double x1 = points[i].x;
                double y1 = points[i].y;

                double x2 = points[idx2].x;
                double y2 = points[idx2].y;

                double x3 = points[idx3].x;
                double y3 = points[idx3].y;

                double x_c, y_c, radius;

                try {
                    findCircle(x1, y1, x2, y2, x3, y3, x_c, y_c, radius);
                    circles.emplace_back(x_c, y_c, radius);  // Store the circle center and radius
                } catch (const std::invalid_argument& e) {
                    // Handle collinear points; you can log or ignore them
                    // For now, we just continue
                    continue;
                }
            }
        }

        return circles;  // Return the vector of circles
    }

    std::vector<cv::Scalar> filterCirclesByRadius(const std::vector<cv::Scalar>& allCircles, double minRadius, double maxRadius) {
        std::vector<cv::Scalar> filteredCircles;

        for (const auto& circle : allCircles) {
            double radius = circle[2]; // The radius is the third element in cv::Scalar
            if (radius >= minRadius && radius <= maxRadius) {
                filteredCircles.push_back(circle);
            }
        }

        return filteredCircles;
    }

    void drawCircles(cv::Mat& image, const std::vector<cv::Scalar>& circles, const cv::Scalar& color, int mode, int radius) {
        for (const auto& circle : circles) {
            // Extract the center position and radius
            cv::Point center(static_cast<int>(circle[0]), static_cast<int>(circle[1]));
            // int radius = static_cast<int>(circle[2]);

            // Draw based on the mode
            switch (mode) {
                case 0: // Draw centers only
                    // Draw the center position as a filled circle
                    cv::circle(image, center, 1, color, cv::FILLED); // Small circle for center
                    break;
                case 1: // Draw circles only
                    // Draw the circle outline
                    cv::circle(image, center, radius, color, 1); // Circle outline
                    break;
                case 2: // Draw both
                    // Draw the center position as a filled circle
                    cv::circle(image, center, 1, color, cv::FILLED); // Small circle for center
                    // Draw the circle outline
                    cv::circle(image, center, radius, color, 1); // Circle outline
                    break;
                default:
                    std::cerr << "Invalid mode selected. Use 0 for centers, 1 for circles, or 2 for both." << std::endl;
                    break;
            }
        }
    }

    cv::Mat applyGaussianBlur(const cv::Mat& inputImage, int kernelSize, double sigmaX) {
        // Check if the input image is empty
        if (inputImage.empty()) {
            std::cerr << "Input image is empty!" << std::endl;
            return cv::Mat();
        }

        // Ensure kernel size is odd and greater than 1
        if (kernelSize % 2 == 0 || kernelSize <= 1) {
            std::cerr << "Kernel size must be an odd number greater than 1!" << std::endl;
            return cv::Mat();
        }

        // Output image
        cv::Mat outputImage;

        // Apply Gaussian blur
        cv::GaussianBlur(inputImage, outputImage, cv::Size(kernelSize, kernelSize), sigmaX);

        return outputImage;
    }

    // Function to calculate point densities and return as cv::Scalar
    std::vector<cv::Scalar> calculatePointDensities(const std::vector<cv::Scalar>& validCircles, double radius) {
        std::vector<cv::Scalar> result;

        for (const auto& circle : validCircles) {
            int count = 0;

            // Check surrounding points within the specified radius
            for (const auto& otherCircle : validCircles) {
                if (&circle != &otherCircle) { // Avoid counting the circle itself
                    double distance = cv::norm(circle - otherCircle);
                    if (distance <= radius) {
                        count++;
                    }
                }
            }

            // Create a cv::Scalar for the point and its density
            cv::Scalar pointDensity(circle[0], circle[1], count);
            result.push_back(pointDensity);
        }

        return result;
    }

    // Function to filter points by density threshold
    std::vector<cv::Scalar> filterPointsByDensity(const std::vector<cv::Scalar>& points, int densityThreshold) {
        std::vector<cv::Scalar> filteredPoints;

        // Filter points based on the density threshold
        for (const auto& point : points) {
            if (point[2] > densityThreshold) { // Check if density is greater than the threshold
                filteredPoints.push_back(point);
            }
        }

        return filteredPoints;
    }

void set_goal(double goal_x, double goal_y, double goal_yaw = 0.0) {
    // Wait for the action server to be available
    if (!action_client_->wait_for_action_server(std::chrono::seconds(5))) {
        RCLCPP_ERROR(this->get_logger(), "NavigateToPose action server not available!");
        return;
    }

    // Create and send the goal
    auto goal_msg = NavigateToPose::Goal();
    goal_msg.pose.pose.position.x = goal_x;
    goal_msg.pose.pose.position.y = goal_y;
    goal_msg.pose.pose.orientation.z = goal_yaw;  // yaw assumed here for simplicity

    auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();

    // Set the callbacks using lambda expressions
    send_goal_options.goal_response_callback = [this](GoalHandleNavigateToPose::SharedPtr goal_handle) {
        if (!goal_handle) {
            RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server.");
        } else {
            RCLCPP_INFO(this->get_logger(), "Goal accepted by server.");
        }
    };

    send_goal_options.result_callback = [this](const rclcpp_action::ClientGoalHandle<NavigateToPose>::WrappedResult & result) {
        // Handle the result of the action here
        switch (result.code) {
            case rclcpp_action::ResultCode::SUCCEEDED:
                RCLCPP_INFO(this->get_logger(), "Goal succeeded!");
                break;
            case rclcpp_action::ResultCode::CANCELED:
                RCLCPP_INFO(this->get_logger(), "Goal was canceled.");
                break;
            case rclcpp_action::ResultCode::ABORTED:
                RCLCPP_ERROR(this->get_logger(), "Goal was aborted.");
                break;
            default:
                RCLCPP_ERROR(this->get_logger(), "Unknown result code.");
                break;
        }
    };

    action_client_->async_send_goal(goal_msg, send_goal_options);
}


    void goal_response_callback(std::shared_future<GoalHandleNavigateToPose::SharedPtr> future) {
        auto goal_handle = future.get();
        if (!goal_handle) {
            RCLCPP_INFO(this->get_logger(), "Goal rejected by server.");
        } else {
            RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result...");
        }
    }

    void result_callback(const GoalHandleNavigateToPose::WrappedResult& result) {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_INFO(this->get_logger(), "Goal reached successfully.");
        } else {
            RCLCPP_INFO(this->get_logger(), "Goal failed with status code: %d", static_cast<int>(result.code));
        }
    }

    cv::Point averagePosition(const std::vector<cv::Scalar>& data) {
        // Check if the vector is empty
        if (data.empty()) {
            return cv::Point(0, 0); // Return a default point if no data is provided
        }

        double sumX = 0.0;
        double sumY = 0.0;

        // Iterate through each scalar in the vector
        for (const auto& scalar : data) {
            sumX += scalar[0]; // x position
            sumY += scalar[1]; // y position
        }

        // Calculate average
        double avgX = sumX / data.size();
        double avgY = sumY / data.size();

        // Return the average as a cv::Point
        return cv::Point(static_cast<int>(avgX), static_cast<int>(avgY));
    }

    void system(){
        int Img_Size, Density_Thresh;
        this->get_parameter("Img Size", Img_Size);
        double Lidar_Min, Lidar_Max, Pursuit_Object_Size, Filtered_Radius, Density_Radius, Temp, Temp2, Temp3;
        this->get_parameter("Lidar Min", Lidar_Min);
        this->get_parameter("Lidar Max", Lidar_Max);
        this->get_parameter("Pursuit Object Size", Pursuit_Object_Size);
        this->get_parameter("Filtered Radius", Filtered_Radius);
        this->get_parameter("Density Radius", Density_Radius);
        this->get_parameter("Density Thresh", Density_Thresh);
        this->get_parameter("Temp", Temp);
        this->get_parameter("Temp2", Temp2);
        this->get_parameter("Temp2", Temp3);

        if(!(latest_laserscan_ == nullptr)){
            // Creates a base image with lidar and the scan range as Red
            cv::Mat baselaser = laserScanToMat(latest_laserscan_, Img_Size, 0);
            cv::Mat process = convertChannels(laserScanToMat(latest_laserscan_, Img_Size, 0, Lidar_Min, Lidar_Max), 3);
            replaceColor(process, cv::Scalar(255,255,255), cv::Scalar(0,0,255));
            blendImages(convertChannels(baselaser.clone(), process), process, baselaser);

            // std::cout << "rsaoechkisaoetnbx" << std::endl;

            // Calculate all potential circles
            if((latest_laserscan_->ranges.size())>=3){
                // std::cout << "Ranges: " << latest_laserscan_->ranges.size() << std::endl;
                    // Generate Circle for Min Max
                cv::circle(baselaser, cv::Point(Img_Size/2, Img_Size/2), (Lidar_Max / latest_laserscan_->range_max) * (Img_Size/2), cv::Scalar(150, 150, 150), 1);
                cv::circle(baselaser, cv::Point(Img_Size/2, Img_Size/2), (Lidar_Min / latest_laserscan_->range_max) * (Img_Size/2), cv::Scalar(100, 100, 100), 1);

                double center, radius, xpos, ypos;
                std::vector<cv::Point> laserscanpoints = laserScanToMatVector(latest_laserscan_, Img_Size, 0, Lidar_Min, Lidar_Max);
                if(laserscanpoints.size()>3){
                    std::vector<cv::Scalar> allCircles = findCircles(laserscanpoints, 4);
                    std::vector<cv::Scalar> tempcircle5 = findCircles(laserscanpoints, 5);
                    std::vector<cv::Scalar> tempcircle6 = findCircles(laserscanpoints, 6);
                    if(!tempcircle5.empty()){allCircles.insert(allCircles.end(), tempcircle5.begin(), tempcircle5.end());}
                    if(!tempcircle6.empty()){allCircles.insert(allCircles.end(), tempcircle6.begin(), tempcircle6.end());}

                    // Filter all potential circles by radius
                    std::vector<cv::Scalar> validCircles = filterCirclesByRadius(allCircles, 0, Filtered_Radius);

                    // Sort all by most dense
                    std::vector<cv::Scalar> circleDensity = calculatePointDensities(validCircles, Density_Radius);

                    // Return by dense
                    std::vector<cv::Scalar> circleDensityFiltered = filterPointsByDensity(circleDensity, Density_Thresh);

                    // Filter densities
                    std::vector<cv::Scalar> filterDensities = calculatePointDensities(circleDensityFiltered, Temp);

                    // Return by dense
                    std::vector<cv::Scalar> filteredDensities = filterPointsByDensity(filterDensities, Temp2);

                    // Process all potential circles to average them out
                    cv::Mat validcircles = baselaser.clone();
                    validcircles.setTo(cv::Scalar(0,0,0));
                    drawCircles(validcircles, filteredDensities, cv::Scalar(255, 255, 0), 0, (Pursuit_Object_Size / latest_laserscan_->range_max) * (Img_Size/2));
                    cv::Mat blurredcircles = applyGaussianBlur(validcircles, 13, 9);
                    cv::floodFill(blurredcircles, cv::Point(1,1), cv::Scalar(255,0,255));

                    cv::Mat potentialCircle = convertChannels(laserScanToMat(latest_laserscan_, Img_Size, 0),3);
                    cv::Mat showthroughcircles = baselaser.clone();
                    showthroughcircles.setTo(cv::Scalar(255,255,255));
                    cv::inRange(blurredcircles.clone(), cv::Scalar(255,0,255), cv::Scalar(255,0,255), blurredcircles);
                    showthroughcircles.setTo(cv::Scalar(0,0,0), blurredcircles);
                    cv::bitwise_not(showthroughcircles.clone(), showthroughcircles);
                    blendImages(potentialCircle.clone(), showthroughcircles, potentialCircle);
                    cv::floodFill(potentialCircle, cv::Point(1,1), cv::Scalar(0,0,0));
                    blendImages(baselaser.clone(), potentialCircle, baselaser);
                    blendImages(baselaser.clone(), validcircles, baselaser);
                }
                // Waypoint
                // cv::circle(baselayer, )

                if(temp){
                cv::Point goal = cv::Point(latest_odom_->pose.pose.position.x + 1, latest_odom_->pose.pose.position.y + 1);//averagePosition(filteredDensities);
                std::cout << goal.x << "   " << goal.y << std::endl;
                set_goal(goal.x, goal.y, 0);
                temp = 0;
                }

                cv::imshow("test5", baselaser);

            }else{std::cout << "Nothing on lidar" << std::endl;}


            cv::waitKey(1);
        }
    }

private:
    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laserscan_sub_;
    std::shared_ptr<sensor_msgs::msg::LaserScan> latest_laserscan_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    std::shared_ptr<nav_msgs::msg::Odometry> latest_odom_;
    rclcpp_action::Client<NavigateToPose>::SharedPtr action_client_;

    bool temp = 1;
};

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<System>());
    rclcpp::shutdown();
    return 0;
}
