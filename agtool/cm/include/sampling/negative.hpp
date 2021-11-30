#pragma once
#include <vector>
#include <iostream>

using namespace std;

pair<vector<int>, vector<int>> _negative_sampling(
    vector<int>& indptr, vector<int>& indices, int num_negatives, int num_items, int num_threads
);