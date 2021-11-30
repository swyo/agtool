#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <filesystem>

#include <omp.h>

#include "sampling/negative.hpp"

using namespace std;

int main(void){
    printf("Test for negative sampling module.\n");
    string line;
    string PATH = static_cast<string>(filesystem::current_path());
    string tmp = PATH + "/dataset/ml-100k/processed/indptr";
    ifstream fin_indptr(PATH + "/dataset/ml-100k/processed/indptr");
    ifstream fin_indices(PATH + "/dataset/ml-100k/processed/indices");
    vector<int> indptr, indices;
    pair<vector<int>, vector<int>> ret;
    while (getline(fin_indptr, line)) indptr.push_back(atoi(line.c_str()));
    while (getline(fin_indices, line)) indices.push_back(atoi(line.c_str()));
    cout << "Size| indptr: " << indptr.size() << ", indices: " << indices.size() << endl;
    int num_items = *max_element(indices.begin(), indices.end()) + 1;
    cout << "num_users: " << indptr.size() - 1 << ",num_items: " << num_items << endl;
    for (int num_threads = 1; num_threads < 7; num_threads++){
        double start = omp_get_wtime();
        ret = _negative_sampling(indptr, indices, 5, num_items, num_threads);
        vector<int> uids = ret.first;
        vector<int> iids = ret.second;
        printf("[negative_sampling] takes %f seconds for [%d] threads\n", omp_get_wtime() - start, num_threads);
    }
    fin_indptr.close();
    fin_indices.close();
    printf("Done.\n");
    return 0;
}
