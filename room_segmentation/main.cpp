/*!
 *****************************************************************
 * \file
 *
 * \note
 * Copyright (c) 2015 \n
 * Fraunhofer Institute for Manufacturing Engineering
 * and Automation (IPA) \n\n
 *
 *****************************************************************
 *
 * \note
 * Project name: Care-O-bot
 * \note
 * ROS stack name: autopnp
 * \note
 * ROS package name: ipa_room_segmentation
 *
 * \author
 * Author: Florian Jordan
 * \author
 * Supervised by: Richard Bormann
 *
 * \date Date of creation: 08.2015
 *
 * \brief
 *
 *
 *****************************************************************
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer. \n
 * - Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. \n
 * - Neither the name of the Fraunhofer Institute for Manufacturing
 * Engineering and Automation (IPA) nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission. \n
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License LGPL as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License LGPL for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License LGPL along with this program.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************/


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <list>
#include <string>
#include <vector>

#include <ipa_room_segmentation/voronoi_segmentation.h>
#include <ipa_room_segmentation/meanshift2d.h>

#include <boost/algorithm/string.hpp>

static bool DEBUG_DISPLAYS=false;

int main(int argc, char** argv)
{
	std::string image_path = cv::samples::findFile("./../../data/recolored_topdown/1LXtFkjw3qL_0.bmp");
    cv::Mat original_img = cv::imread(image_path, cv::IMREAD_COLOR);

	cv::uint8_t white_val = 255;
	cv::uint8_t black_val = 0;

	for (int j = 0; j < original_img.cols; j++) {
		for (int i = 0; i < original_img.rows; i++) {
            if (original_img.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 255, 255))
                original_img.at<cv::Vec3b>(i, j) = cv::Vec3b(black_val, black_val, black_val);
            if (original_img.at<cv::Vec3b>(i, j) == cv::Vec3b(128, 128, 128))
                original_img.at<cv::Vec3b>(i, j) = cv::Vec3b(white_val, white_val, white_val);
		}
	}

	// original_img.convertTo(original_img, CV_8UC1);
	cv::Mat img;
	cv::cvtColor(original_img, img, cv::COLOR_BGR2GRAY);

	const float map_resolution = 0.1;
	const cv::Point2d map_origin(const float map_origin_position_x=0.0, const float map_origin_position_y=0.0);
    const double robot_radius = 0.3;
    double room_lower_limit_voronoi_ = 0.1;	//1.53;
	double room_upper_limit_voronoi_ = 1000000.;	//120.0;
	int voronoi_neighborhood_index_ = 280;
	int max_iterations_ = 16;
	double min_critical_point_distance_factor_ = 0.5; //1.6;
	double max_area_for_merging_ = 12.5;
	bool display_segmented_map_ = true;
	bool publish_segmented_map_ = true;

	//segment the given map
	cv::Mat segmented_map;
	VoronoiSegmentation voronoi_segmentation; //voronoi segmentation method
	voronoi_segmentation.segmentMap(img, segmented_map, map_resolution, room_lower_limit_voronoi_, room_upper_limit_voronoi_,
		voronoi_neighborhood_index_, max_iterations_, min_critical_point_distance_factor_, max_area_for_merging_, (display_segmented_map_&&DEBUG_DISPLAYS));

	// get the min/max-values and the room-centers
	// compute room label codebook
	std::map<int, size_t> label_vector_index_codebook; // maps each room label to a position in the rooms vector
	size_t vector_index = 0;
	for (int v = 0; v < segmented_map.rows; ++v)
	{
		for (int u = 0; u < segmented_map.cols; ++u)
		{
			const int label = segmented_map.at<int>(v, u);
			if (label > 0 && label < 65280) // do not count walls/obstacles or free space as label
			{
				if (label_vector_index_codebook.find(label) == label_vector_index_codebook.end())
				{
					label_vector_index_codebook[label] = vector_index;
					vector_index++;
				}
			}
		}
	}
	//min/max y/x-values vector for each room. Initialized with extreme values
	std::vector<int> min_x_value_of_the_room(label_vector_index_codebook.size(), 100000000);
	std::vector<int> max_x_value_of_the_room(label_vector_index_codebook.size(), 0);
	std::vector<int> min_y_value_of_the_room(label_vector_index_codebook.size(), 100000000);
	std::vector<int> max_y_value_of_the_room(label_vector_index_codebook.size(), 0);
	//vector of the central Point for each room, initially filled with Points out of the map
	std::vector<int> room_centers_x_values(label_vector_index_codebook.size(), -1);
	std::vector<int> room_centers_y_values(label_vector_index_codebook.size(), -1);
	//***********************Find min/max x and y coordinate and center of each found room********************
	//check y/x-value for every Pixel and make the larger/smaller value to the current value of the room
	for (int y = 0; y < segmented_map.rows; ++y)
	{
		for (int x = 0; x < segmented_map.cols; ++x)
		{
			const int label = segmented_map.at<int>(y, x);
			if (label > 0 && label < 65280) //if Pixel is white or black it is no room --> doesn't need to be checked
			{
				const int index = label_vector_index_codebook[label];
				min_x_value_of_the_room[index] = std::min(x, min_x_value_of_the_room[index]);
				max_x_value_of_the_room[index] = std::max(x, max_x_value_of_the_room[index]);
				max_y_value_of_the_room[index] = std::max(y, max_y_value_of_the_room[index]);
				min_y_value_of_the_room[index] = std::min(y, min_y_value_of_the_room[index]);
			}
		}
	}
	//get centers for each room
//	for (size_t idx = 0; idx < room_centers_x_values.size(); ++idx)
//	{
//		if (max_x_value_of_the_room[idx] != 0 && max_y_value_of_the_room[idx] != 0 && min_x_value_of_the_room[idx] != 100000000 && min_y_value_of_the_room[idx] != 100000000)
//		{
//			room_centers_x_values[idx] = (min_x_value_of_the_room[idx] + max_x_value_of_the_room[idx]) / 2;
//			room_centers_y_values[idx] = (min_y_value_of_the_room[idx] + max_y_value_of_the_room[idx]) / 2;
//			cv::circle(segmented_map, cv::Point(room_centers_x_values[idx], room_centers_y_values[idx]), 2, cv::Scalar(200*256), CV_FILLED);
//		}
//	}
	// use distance transform and mean shift to find good room centers that are reachable by the robot
	// first check whether a robot radius shall be applied to obstacles in order to exclude room center points that are not reachable by the robot
	cv::Mat segmented_map_copy = segmented_map;
	cv::Mat connection_to_other_rooms = cv::Mat::zeros(segmented_map.rows, segmented_map.cols, CV_8UC1);	// stores for each pixel whether a path to another rooms exists for a robot of size robot_radius
	if (robot_radius > 0.0)
	{
		// consider robot radius for exclusion of non-reachable points
		segmented_map_copy = segmented_map.clone();
		cv::Mat map_8u, eroded_map;
		segmented_map_copy.convertTo(map_8u, CV_8UC1, 1., 0.);
		int number_of_erosions = (robot_radius / map_resolution);
		cv::erode(map_8u, eroded_map, cv::Mat(), cv::Point(-1, -1), number_of_erosions);
		for (int v=0; v<segmented_map_copy.rows; ++v)
			for (int u=0; u<segmented_map_copy.cols; ++u)
				if (eroded_map.at<uchar>(v,u) == 0)
					segmented_map_copy.at<int>(v,u) = 0;

		// compute connectivity of remaining accessible room cells to other rooms
		bool stop = false;
		while (stop == false)
		{
			stop = true;
			for (int v=1; v<segmented_map_copy.rows-1; ++v)
			{
				for (int u=1; u<segmented_map_copy.cols-1; ++u)
				{
					// skip already identified cells
					if (connection_to_other_rooms.at<uchar>(v,u) != 0)
						continue;

					// only consider cells labeled as a room
					const int label = segmented_map_copy.at<int>(v,u);
					if (label <= 0 || label >= 65280)
						continue;

					for (int dv=-1; dv<=1; ++dv)
					{
						for (int du=-1; du<=1; ++du)
						{
							if (dv==0 && du==0)
								continue;
							const int neighbor_label = segmented_map_copy.at<int>(v+dv,u+du);
							if (neighbor_label>0 && neighbor_label<65280 && (neighbor_label!=label || (neighbor_label==label && connection_to_other_rooms.at<uchar>(v+dv,u+du)==255)))
							{
								// either the room cell has a direct border to a different room or the room cell has a neighbor from the same room label with a connecting path to another room
								connection_to_other_rooms.at<uchar>(v,u) = 255;
								stop = false;
							}
						}
					}
				}
			}
		}
	}
	// compute the room centers
	MeanShift2D ms;
	for (std::map<int, size_t>::iterator it = label_vector_index_codebook.begin(); it != label_vector_index_codebook.end(); ++it)
	{
		int trial = 1; 	// use robot_radius to avoid room centers that are not accessible by a robot with a given radius
		if (robot_radius <= 0.)
			trial = 2;

		for (; trial <= 2; ++trial)
		{
			// compute distance transform for each room on the room cells that have some connection to another room (trial 1) or just on all cells of that room (trial 2)
			const int label = it->first;
			int number_room_pixels = 0;
			cv::Mat room = cv::Mat::zeros(segmented_map_copy.rows, segmented_map_copy.cols, CV_8UC1);
			for (int v = 0; v < segmented_map_copy.rows; ++v)
				for (int u = 0; u < segmented_map_copy.cols; ++u)
					if (segmented_map_copy.at<int>(v, u) == label && (trial==2 || connection_to_other_rooms.at<uchar>(v,u)==255))
					{
						room.at<uchar>(v, u) = 255;
						++number_room_pixels;
					}
			if (number_room_pixels == 0)
				continue;
			cv::Mat distance_map; //variable for the distance-transformed map, type: CV_32FC1
			cv::distanceTransform(room, distance_map, cv::DIST_L2, 5);
			// find point set with largest distance to obstacles
			double min_val = 0., max_val = 0.;
			cv::minMaxLoc(distance_map, &min_val, &max_val);
			std::vector<cv::Vec2d> room_cells;
			for (int v = 0; v < distance_map.rows; ++v)
				for (int u = 0; u < distance_map.cols; ++u)
					if (distance_map.at<float>(v, u) > max_val * 0.95f)
						room_cells.push_back(cv::Vec2d(u, v));
			if (room_cells.size()==0)
				continue;
			// use meanshift to find the modes in that set
			cv::Vec2d room_center = ms.findRoomCenter(room, room_cells, map_resolution);
			const int index = it->second;
			room_centers_x_values[index] = room_center[0];
			room_centers_y_values[index] = room_center[1];

			if (room_cells.size() > 0)
				break;
		}
	}

	// convert the segmented map into an indexed map which labels the segments with consecutive numbers (instead of arbitrary unordered labels in segmented map)
	cv::Mat indexed_map = segmented_map.clone();
	for (int y = 0; y < segmented_map.rows; ++y)
	{
		for (int x = 0; x < segmented_map.cols; ++x)
		{
			const int label = segmented_map.at<int>(y,x);
			if (label > 0 && label < 65280)
				indexed_map.at<int>(y,x) = label_vector_index_codebook[label]+1;//start value from 1 --> 0 is reserved for obstacles
		}
	}

	if (display_segmented_map_ == true)
	{
		// colorize the segmented map with the indices of the room_center vector
		cv::Mat color_segmented_map = indexed_map.clone();
		color_segmented_map.convertTo(color_segmented_map, CV_8U);
		cv::cvtColor(color_segmented_map, color_segmented_map, cv::COLOR_GRAY2BGR);
		for(size_t i = 1; i <= room_centers_x_values.size(); ++i)
		{
			//choose random color for each room
			const cv::Vec3b color((rand() % 250) + 1, (rand() % 250) + 1, (rand() % 250) + 1);
			for(size_t v = 0; v < indexed_map.rows; ++v)
				for(size_t u = 0; u < indexed_map.cols; ++u)
					if(indexed_map.at<int>(v,u) == i)
						color_segmented_map.at<cv::Vec3b>(v,u) = color;
		}
//		cv::Mat disp = segmented_map.clone();
		for (size_t index = 0; index < room_centers_x_values.size(); ++index)
			cv::circle(color_segmented_map, cv::Point(room_centers_x_values[index], room_centers_y_values[index]), 2, cv::Scalar(256), cv::FILLED);

		cv::namedWindow("segmentation", cv::WINDOW_NORMAL);
		cv::resizeWindow("segmentation", 1000, 1000);
		cv::imshow("segmentation", color_segmented_map);
		cv::waitKey();
	}
}
