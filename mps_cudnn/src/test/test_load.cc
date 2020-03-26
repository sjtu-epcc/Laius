/*
 * File: /Users/gema/Projects/paper/reference/mps/src/test/test_load.cc
 * Project: /Users/gema/Projects/paper/reference/mps
 * Created Date: Saturday, January 5th 2019, 4:28:57 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Saturday, January 5th 2019, 10:54:52 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include "check.hpp"
// #include <gflags/gflags.h>
#include <unordered_map>
#include <fstream>

void CsvToMap(std::ifstream &input, std::unordered_map<std::vector<int>, double, container_hash<std::vector<int>>> &dict)
{
    std::string line_input;
    double elapse;
    int cnt = 0;
    int cnt_dup = 0;
    while (getline(input, line_input))
    {
        cnt++;
        std::vector<int> int_key;
        std::vector<std::string> str_key;
        std::cout << "------Reading Model------" << std::endl;
        std::stringstream key_stream(line_input);
        std::string key_s;
        while (getline(key_stream, key_s, ','))
        {
            str_key.push_back(key_s);
        }
        for (size_t i = 0; i != str_key.size(); i++)
        {
            if (i == (str_key.size() - 1))
            {
                elapse = atof(str_key[i].c_str());
                break;
            }
            int_key.push_back(atoi(str_key[i].c_str()));
        }
        if (!dict.emplace(int_key, elapse).second)
        {
            cnt_dup++;
            std::cout << "------" << cnt_dup << "-th Duplicate Key------" << std::endl;
            dict[int_key] = elapse;
        }
        std::cout << "------" << cnt << "-th Key------" << std::endl;
    }
}

int main(int argc, char const *argv[])
{
    std::ifstream input;
    std::unordered_map<std::vector<int>, double, container_hash<std::vector<int>>> output;
    input.open("../../data/cudnnActivationForward_b.csv", std::ios::in);
    CsvToMap(input, output);
    std::unordered_map<std::vector<int>, double, container_hash<std::vector<int>>>::iterator it;
    for (it = output.begin(); it != output.end(); it++)
    {
        std::cout << "percentage " << it->first.back() << " time: " << it->second << std::endl;
    }
    return 0;
}
