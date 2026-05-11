#pragma once

#include <vector>
#include <string>
using namespace std;

void load_csv(const string& filename, vector<double>& data, int rows, int cols);

void save_csv(const vector<double>& data, int rows, int cols, const string& filename);

void ruler(int width = 60);