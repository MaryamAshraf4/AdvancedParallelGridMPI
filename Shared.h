#pragma once

#include <vector>
#include <string>

void load_csv(const std::string& filename,
    std::vector<double>& data,
    int rows, int cols);

void save_csv(const std::vector<double>& data,
    int rows, int cols,
    const std::string& filename);

void ruler(int width = 60);