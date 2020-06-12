#!/bin/bash

./2D_feature_tracking SHITOMASI | grep 'MP.7 Performance Evaluation 1' -A4 -B4
./2D_feature_tracking HARRIS | grep 'MP.7 Performance Evaluation 1' -A4 -B4
./2D_feature_tracking FAST | grep 'MP.7 Performance Evaluation 1' -A4 -B4
./2D_feature_tracking BRISK | grep 'MP.7 Performance Evaluation 1' -A4 -B4
./2D_feature_tracking ORB | grep 'MP.7 Performance Evaluation 1' -A4 -B4
./2D_feature_tracking AKAZE | grep 'MP.7 Performance Evaluation 1' -A4 -B4
./2D_feature_tracking SIFT | grep 'MP.7 Performance Evaluation 1' -A4 -B4

