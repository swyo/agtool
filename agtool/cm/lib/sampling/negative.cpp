#include <cassert>
#include <random>
#include <iterator>
#include <algorithm>
#include <omp.h>

#include <sampling/negative.hpp>

using namespace std;

pair<vector<int>, vector<int>> _negative_sampling(
    vector<int>& indptr, vector<int>& indices, int num_negatives, int num_items, int num_threads
){
    vector<int> uids;
    vector<int> iids;
    vector<int> items;
    int DUMMY = 100000;
    int total_num_negatives = 0;
    for (int i = 0; i < num_items; i++) items.push_back(i);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(indptr.size()) - 1; i++){
        int num_positives = indptr[i + 1] - indptr[i];
        vector<int> positives(num_positives, DUMMY);
        vector<int> negatives;
        vector<int> diff;
        for (int j = 0; j < num_positives; j++)
            positives[j] = indices[indptr[i] + j];
        set_difference(items.begin(), items.end(), positives.begin(), positives.end(), inserter(diff, diff.begin()));
        sample(diff.begin(), diff.end(), std::back_inserter(negatives), num_negatives, mt19937{random_device{}()});
        assert(negatives.size() <= num_negatives);
        #pragma omp critical
            for (auto e: negatives){
                iids.push_back(e);
                uids.push_back(i);
            }   
            total_num_negatives += negatives.size();
    }
    assert(uids.size() == iids.size());
    return make_pair(uids, iids);
}
